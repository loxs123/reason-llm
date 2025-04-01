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
from datasets import load_dataset,load_from_disk
from trl.data_utils import maybe_apply_chat_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
import ray
import glob

from reason_llm.utils import apply_lora, get_per_token_logps, remove_stutter
from reason_llm.config import *
from reason_llm.reward_fn import *

def load_data_from_disk_or_hf(data_name):
    if os.path.exists(data_name):
        return load_from_disk(data_name)
    return load_dataset(data_name)

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
            enable_prefix_caching=True,
            dtype='bfloat16',
        )
        self.tokenizer = copy.deepcopy(self.llm.get_tokenizer())

    def generate(self, prompts, sampling_params):
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        outputs = self.llm.generate(formatted_prompts, sampling_params=sampling_params, use_tqdm=False)
        return [
            prompt + [{"role": "assistant", "content": output.outputs[0].text}]
            # prompt + [{"role": "assistant", "content": remove_stutter(output.outputs[0].text)}]
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
        self.log_config()
        self.train_data = load_data_from_disk_or_hf(TRAIN_DATASET)['train'].shuffle(seed=42)
        self.test_data = [load_data_from_disk_or_hf(ds_name) for ds_name in TEST_DATASETS]
        print('Train Set size: ', len(self.train_data))
        self.init_vllm_workers()
    
    def log_config(self):
        with open(config_file) as f:
            print('#' * 10, 'CONFIG_START', '#' * 10)
            print(f.read())
            print('#' * 10, 'CONFIG_END', '#' * 10)
    
    def _find_buffer_id(self):
        buffer_file_pattern = os.path.join(data_dir, 'buffer*.json')
        buffer_files = glob.glob(buffer_file_pattern)
        buffer_files.sort(reverse = True)

        if len(buffer_files) == 0:
            buffer_id = 0
        else:
            buffer_id = int(re.findall('buffer(.*?).json', buffer_files[0])[0]) + 1
        
        return buffer_id
        
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
        
    def _generate(self, prompts, sampling_params):
        num_workers = len(self.workers)
        if num_workers == 0:
            raise RuntimeError("No available VLLM workers!")

        # 任务分配
        chunk_size = (len(prompts) + num_workers - 1) // num_workers
        prompt_chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]

        # 并行推理
        futures = [worker.generate.remote(chunk, sampling_params) for worker, chunk in zip(self.workers, prompt_chunks)]
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

    def log_info(self, dataset_name):
        print(f'[{dataset_name}] Average Acc: {mean(self.acc_ave)}，'
            f'Major Acc: {mean(self.acc_major)}，'
            f'Average Reward: {mean(self.reward)}，'
            f'Average Length: {mean(self.length)}')

    def clear_info(self):
        self.acc_ave.clear()
        self.acc_major.clear()
        self.reward.clear()
        self.length.clear()

    def _to_buffer(self, buffer_msgs, buffer_sols, mode='train'):

        if mode == 'train':
            sampling_params = SamplingParams(
                temperature=1.0,
                max_tokens=TRAIN_MAX_GENERATE_LEN,
            )
        else:
            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1,
                max_tokens=EVAL_MAX_GENERATE_LEN,
            )

        num_generations = NUM_GENERATIONS if mode == 'train' else 1

        # INT_NUM
        msgs = [msg for _msgs in buffer_msgs for msg in _msgs]
        msgs = self._generate(msgs, sampling_params)

        buffers = []
        for j in range(0, len(msgs), num_generations):
            rewards, acc_rate, major = self._group_reward_fn(
                completions=msgs[j:j+num_generations],
                solution=buffer_sols[j//num_generations],
            )
            self.reward.extend(rewards)
            self.acc_ave.append(acc_rate)
            self.acc_major.append(major)
            self.length.append(ave_length(msgs[j:j+num_generations], self.tokenizer))
            rewards = np.array(rewards)
            # 
            if acc_rate <= 0.01 or acc_rate >= 0.99:
                continue
            # advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4) # grpo
            advantages = rewards - rewards.mean() # 

            for msg, advantage,reward in zip(msgs[j:j+num_generations], advantages, rewards):
                buffers.append({
                    "completion": msg,
                    "advantage": advantage.item(),
                    "reward": reward.item(),
                    "label": buffer_sols[j // num_generations],
                })

        return buffers

    def compute_logp(self):
        print("\n[ComputeLogp Stage]")
        old_model_path = os.path.join(model_dir, "merge")
        if not os.path.exists(old_model_path):
            old_model_path = model_dir
        batch_size = per_device_train_batch_size
        
        model_dict = {"old_per_token_logps": old_model_path}

        for key, model_id in model_dict.items():
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda')
            
            for i in range(0, len(self.buffers), batch_size):
                prompts_text = [maybe_apply_chat_template({'messages': example['completion']}, self.tokenizer)["text"] \
                                 for example in self.buffers[i:i+batch_size]]
                prompt_inputs = self.tokenizer(
                    prompts_text, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )['input_ids'].to('cuda')
                with torch.inference_mode():
                    logps = get_per_token_logps(model, prompt_inputs).cpu().tolist()

                for j in range(len(logps)):
                    self.buffers[i + j][key] = logps[j]

            del model
            gc.collect()
            torch.cuda.empty_cache()  # 清空未被使用的显存缓存


    def generate_samples(self):
        print("\n[Sample Stage]")
        
        self.buffers = []
        current_msgs = []
        current_sols = []
        int_num_sample = MAX_NUM_SEQ // NUM_GENERATIONS
        while len(self.buffers) < INT_NUM:
            _row = self.train_data[self.train_idx % len(self.train_data)]
            row = {k.lower(): v for k,v in _row.items()}

            current_msgs.append(build_msgs(row, dataset=TRAIN_DATASET, num_generations = NUM_GENERATIONS))
            current_sols.append(build_sol(row, dataset=TRAIN_DATASET))

            if (self.train_idx + 1) % int_num_sample == 0:
                self.buffers += self._to_buffer(current_msgs, current_sols)
                current_msgs.clear()
                current_sols.clear()
            self.train_idx += 1

        self.log_info(TRAIN_DATASET)
        self.buffers = self.buffers[:INT_NUM]
        print('End Train Idx:', self.train_idx)

    def save_samples(self,):
        data = self.buffers
        random.shuffle(data) # 打乱数据
        self.clear_info()
        buffer_id = self._find_buffer_id()
        buffer_file = os.path.join(data_dir, 'buffer%05d.json'%buffer_id)
        with open(buffer_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.buffers.clear()

    def del_vllm_workers(self):
        for worker in self.workers:
            ray.kill(worker)
        self.workers = []
    
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        gc.collect()

    def train_model(self):
        print("\n[Train Stage]")
        if GPU_NUM == 1:
            os.system(f'CUDA_VISIBLE_DEVICES={GPU} accelerate launch'
                      f' --config_file "{current_dir}/reason_llm/ds_cfgs/deepspeed_zero2.yaml" '
                      f'"{current_dir}/reason_llm/grpo_trainer.py"')
        else:
            os.system(f'CUDA_VISIBLE_DEVICES={GPU} accelerate launch'
                      f' --config_file "{current_dir}/reason_llm/ds_cfgs/deepspeed_zero3.yaml" '
                      f'"{current_dir}/reason_llm/grpo_trainer.py"')

    def test_model(self):
        print("\n[Test Stage]")
        for ds, ds_name in zip(self.test_data, TEST_DATASETS):
            current_msgs = []
            current_sols = []

            self.clear_info()
            int_num_sample = MAX_NUM_SEQ
            if 'train' in ds: ds = ds['train']
            elif 'test' in ds: ds = ds['test']

            for i in range(len(ds)):
                _row = ds[i]
                row = {k.lower(): v for k, v in _row.items()}
                current_msgs.append(build_msgs(row, dataset=ds_name, num_generations = 1))
                current_sols.append(build_sol(row, dataset=ds_name))

                if (i + 1) % int_num_sample == 0:
                    self._to_buffer(current_msgs, current_sols, 'test')
                    current_msgs.clear()
                    current_sols.clear()

            if len(current_msgs) > 0:
                self._to_buffer(current_msgs, current_sols, 'test')

            self.log_info(ds_name)
            self.clear_info()
    
    def run_cycle(self):
        self.generate_samples()
        self.del_vllm_workers()
        self.compute_logp()
        self.save_samples()
        self.train_model() # lora：base model_dir lora lora
        self.init_vllm_workers() # merge : base + lora
        self.test_model()