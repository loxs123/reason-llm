import os
import json
import csv
import copy
from tqdm import tqdm
import gc
import numpy as np
import torch
from vllm import LLM, SamplingParams
import random
import time
import re
from datasets import load_dataset
from trl.data_utils import maybe_apply_chat_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
import ray

from reason_llm.utils import apply_lora, get_per_token_logps, remove_stutter
from reason_llm.config import *
from reason_llm.reward_fn import *

def ave_length(msgs, tokenizer):
    l = 0
    for m in msgs:
        l += len(tokenizer.tokenize(m[-1]['content']))
    return l / len(msgs)

def mean(l):
    return sum(l) / len(l)

# 定义 VLLM Worker 远程类
@ray.remote(num_gpus=PER_VLLM_GPU)  # 每个 VLLM 实例占 1 张 GPU
class VLLMWorker:
    def __init__(self, model_dir, gpu_ids):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        self.llm = LLM(
            model_dir,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            tensor_parallel_size=PER_VLLM_GPU,
            max_num_seqs=MAX_NUM_SEQ // len(VLLM_CONFIG),
            gpu_memory_utilization=0.90,
        )
        self.tokenizer = copy.deepcopy(self.llm.get_tokenizer())

    def generate(self, prompts):
        sampling_params = SamplingParams(
            temperature=1.0,
            min_p=0.01,
            max_tokens=MAX_MODEL_LEN,
            skip_special_tokens=True
        )
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        outputs = self.llm.generate(formatted_prompts, sampling_params=sampling_params)
        return [
            prompt + [{"role": "assistant", "content": remove_stutter(output.outputs[0].text)}]
            for prompt, output in zip(prompts, outputs)
        ]

class TrainingSamplingCoordinator:
    def __init__(self):
        self.train_idx = START_TRAIN_IDX
        self.acc_ave = []
        self.reward = []
        self.acc_major = []
        self.length = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.train_data = load_dataset(TRAIN_DATASET)['train'].shuffle(seed=42)
        self.test_data = load_dataset(TEST_DATASET)['train']
        print('训练集数据集总量：', len(self.train_data))
        print('测试集数据集总量：', len(self.test_data))
        self.init_vllm_workers()
        
    def init_vllm_workers(self):
        """创建多个 VLLM Worker"""
        apply_lora(model_dir)
        workers = []
        model_path = os.path.join(model_dir, "merge")
        if not os.path.exists(model_path):
            model_path = model_dir

        for gpu_ids in VLLM_CONFIG:
            worker = VLLMWorker.remote(model_path, gpu_ids)
            workers.append(worker)
        self.workers = workers
        
    def _generate(self, prompts):
        num_workers = len(self.workers)
        if num_workers == 0:
            raise RuntimeError("No available VLLM workers!")

        # 任务分配
        chunk_size = (len(prompts) + num_workers - 1) // num_workers
        prompt_chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]

        # 并行推理
        futures = [worker.generate.remote(chunk) for worker, chunk in zip(self.workers, prompt_chunks)]
        results = ray.get(futures)  # 获取所有结果

        # 保证输出顺序
        final_results = []
        for res in results:
            final_results.extend(res)

        return final_results

    def _group_reward_fn(self, completions, solution):
        r1, mj = accuracy_reward(completions, solution)
        r2 = format_reward(completions)

        r = [_r1 * ACCURACY_WEIGHT + _r2 * FORMAT_WEIGHT for _r1, _r2 in zip(r1, r2)]
        return r, sum(r1) / len(r1), mj

    def log_info(self, cur_msgs=None):

        if cur_msgs is not None:
            print(f'平均正确率：{mean(self.acc_ave)}，'
                f'多数投票正确率：{mean(self.acc_major)}，'
                f'平均奖励：{mean(self.reward)}，'
                f'平均长度：{mean(self.length)}，'
                f'收集轨迹数量：{len(cur_msgs)}')
        else:
            print(f'平均正确率：{mean(self.acc_ave)}，'
                f'多数投票正确率：{mean(self.acc_major)}，'
                f'平均奖励：{mean(self.reward)}，'
                f'平均长度：{mean(self.length)}')
    
    def clear_info(self):
        self.acc_ave.clear()
        self.acc_major.clear()
        self.reward.clear()
        self.length.clear()

    def _to_buffer(self, buffer_msgs, buffer_sols):

        # INT_NUM
        msgs = [msg for _msgs in buffer_msgs for msg in _msgs]
        msgs = self._generate(msgs)

        buffers = []
        for j in range(0, len(msgs), NUM_GENERATIONS):

            # 处理结果
            rewards, acc_rate, major = self._group_reward_fn(
                completions=msgs[j:j+NUM_GENERATIONS],
                solution=buffer_sols[j//NUM_GENERATIONS],
            )
            self.reward.extend(rewards)
            self.acc_ave.append(acc_rate)
            self.acc_major.append(major)
            self.length.append(ave_length(msgs[j:j+NUM_GENERATIONS], self.tokenizer))
            rewards = np.array(rewards)
            if rewards.std() < 0.1: # 技巧1 
                continue
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4) # grpo

            for msg, advantage,reward in zip(msgs[j:j+NUM_GENERATIONS], advantages, rewards):
                if advantage > 0.0: # 技巧2 只训练正样本
                    buffers.append({
                        "completion": msg,
                        "advantage": advantage.item(),
                        "reward": reward.item(),
                        "label": buffer_sols[j // NUM_GENERATIONS],
                    })

        return buffers

    def generate_samples(self):
        print("\n--- 开始采样阶段 ---")

        print('开始采样下标：', self.train_idx)

        self.buffers = []

        current_msgs = []
        current_sols = []
        
        int_num_sample = MAX_NUM_SEQ // NUM_GENERATIONS
        while len(self.buffers) < INT_NUM:
            _row = self.train_data[self.train_idx % len(self.train_data)]
            row = {k.lower(): v for k,v in _row.items()}

            current_msgs.append(build_msgs(row, mode='train'))
            current_sols.append(build_sol(row, mode='train'))

            if (self.train_idx + 1) % int_num_sample == 0:
                self.buffers += self._to_buffer(current_msgs, current_sols)
                current_msgs.clear()
                current_sols.clear()
                self.log_info(self.buffers)
            self.train_idx += 1
        self.buffers = self.buffers[:INT_NUM]
        print('结束采样下标：', self.train_idx)

    def save_samples(self,):
        data = self.buffers * REP_NUM
        random.shuffle(data) # 打乱数据
        self.log_info()
        self.clear_info()
        with open(buffer_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.buffers.clear()
        print(f"已写入{len(data)}条样本到缓冲区")
    
    def del_vllm_workers(self):
        for worker in self.workers:
            ray.kill(worker)
        self.workers = []
    
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
        gc.collect()

    def train_model(self):
        print("\n--- 开始训练阶段 ---")
        if GPU_NUM == 1:
            # for single gpu use deepspeed_zero2
            os.system(f'CUDA_VISIBLE_DEVICES={GPU} accelerate launch --config_file "{current_dir}/reason_llm/ds_cfgs/deepspeed_zero2.yaml" "{current_dir}/reason_llm/grpo_trainer.py"')
        else:
            # for multi gpu use deepspeed_zero3
            os.system(f'CUDA_VISIBLE_DEVICES={GPU} accelerate launch --config_file "{current_dir}/reason_llm/ds_cfgs/deepspeed_zero3.yaml" "{current_dir}/reason_llm/grpo_trainer.py"')

    def test_model(self):
        print("\n--- 开始测试阶段 ---")
        current_msgs = []
        current_sols = []

        self.clear_info()
        int_num_sample = MAX_NUM_SEQ // NUM_GENERATIONS
        for i in range(len(self.test_data)):
            _row = self.test_data[i]
            row = {k.lower(): v for k, v in _row.items()}
            current_msgs.append(build_msgs(row, mode='test'))
            current_sols.append(build_sol(row, mode='test'))

            if (i + 1) % int_num_sample == 0:
                self._to_buffer(current_msgs, current_sols)
                current_msgs.clear()
                current_sols.clear()
                self.log_info()

        if len(current_msgs) > 0:
            self._to_buffer(current_msgs, current_sols)

        self.log_info()
        self.clear_info()
    
    def compute_logp(self):
        ...
        # TODO: BUG - The computed log probability (logp) differs from the Trainer's results.
        # This discrepancy has a substantial impact on the final experimental outcomes(reward)
        # ref_model_path = model_dir
        # old_model_path = os.path.join(model_dir, "merge")
        # if not os.path.exists(old_model_path):
        #     old_model_path = model_dir
        # batch_size = 4
        
        # if BETA > 0.0:
        #     model_dict = {"old_per_token_logps": old_model_path, "ref_per_token_logps": ref_model_path}
        # else:
        #     model_dict = {"old_per_token_logps": old_model_path}

        # for key, model_id in model_dict.items():
        #     model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda')
            
        #     for i in tqdm(range(0, len(self.buffers), batch_size)):
        #         prompts_text = [maybe_apply_chat_template({'messages': example['completion']}, self.tokenizer)["text"] \
        #                          for example in self.buffers[i:i+batch_size]]
        #         prompt_inputs = self.tokenizer(
        #             prompts_text, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
        #         )['input_ids'].to('cuda')
        #         with torch.inference_mode():
        #             logps = get_per_token_logps(model, prompt_inputs).cpu().tolist()

        #         for j in range(len(logps)):
        #             self.buffers[i + j][key] = logps[j]

        #     del model
        #     gc.collect()
        #     torch.cuda.empty_cache()  # 清空未被使用的显存缓存

    def run_cycle(self):
        self.generate_samples()
        self.del_vllm_workers()
        self.compute_logp()
        self.save_samples()
        self.train_model() # lora：base model_dir lora lora
        self.init_vllm_workers() # merge : base + lora
        self.test_model()
