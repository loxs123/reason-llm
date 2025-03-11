import os
import json
import csv
import copy
import gc
import numpy as np
import torch
from vllm import LLM, SamplingParams
import random
import time
import re
from datasets import load_dataset

from reason_llm.utils import apply_lora
from reason_llm.config import *
from reason_llm.reward_fn import *

with open(system_setting_file) as f:
    d = f.read()
    system_settings = re.findall(r'```text\n(.*?)\n```', d, re.S)

assert MAX_NUM_SEQ % len(system_settings) == 0

def ave_length(msgs, tokenizer):
    l = 0
    for m in msgs:
        l += len(tokenizer.tokenize(m[-1]['content']))
    return l / len(msgs)

def mean(l):
    return sum(l) / len(l)

class TrainingSamplingCoordinator:
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.cur_data_idx = 0
        self.acc_ave = []
        self.reward = []
        self.acc_major = []
        self.length = []

        self.data = load_dataset(DATASET)['train']
        print('数据集总量：', len(self.data))
        self.load_model()
        
    def load_model(self):
        """加载或重新加载模型"""
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            gc.collect()
            torch.cuda.empty_cache()

        self.llm = self._create_llm_instance()
        self.tokenizer = copy.deepcopy(self.llm.get_tokenizer())

    def _create_llm_instance(self):
        """创建LLM实例"""

        apply_lora(model_dir)
        model_path = os.path.join(model_dir, "merge")

        if not os.path.exists(model_path):
            model_path = model_dir
        
        return LLM(
            model_path,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            tensor_parallel_size=GPU_NUM,
            max_num_seqs=MAX_NUM_SEQ,
            gpu_memory_utilization=0.90,
        )

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
        batch_prompts = [msg for msgs in buffer_msgs for msg in msgs]
        sample_num = len(buffer_msgs[0])
        completed_batch = self._generate_batch(batch_prompts)

        cur_msgs = []
        for j in range(0, len(completed_batch), sample_num):

            # 处理结果
            rewards, acc_rate, major = self._group_reward_fn(
                completions=completed_batch[j:j+sample_num],
                solution=buffer_sols[j//sample_num],
            )
            self.reward.extend(rewards)
            self.acc_ave.append(acc_rate)
            self.acc_major.append(major)
            self.length.append(ave_length(completed_batch[j:j+sample_num], self.tokenizer))
            rewards = np.array(rewards)
            if rewards.std() < 0.1: # 技巧1 
                continue
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4) # grpo

            for msg, advantage,reward in zip(completed_batch[j:j+sample_num], advantages, rewards):
                if advantage > 0.0: # 技巧2 只训练正样本
                    for _ in range(REP_NUM):
                        cur_msgs.append({
                            "completion": msg,
                            "advantage": advantage.item(),
                            "reward": reward.item(),
                            "label": buffer_sols[j // sample_num],
                        })

        return cur_msgs

    def generate_samples(self):
        print("\n--- 开始采样阶段 ---")

        print('开始采样下标：', self.cur_data_idx)
        data_size = len(self.data)

        buffer_msgs = []
        buffer_sols = []
        cur_msgs = []
        
        i = 0
        while len(cur_msgs) < INT_NUM * REP_NUM:
            row = self.data[(self.cur_data_idx + i) % data_size]
            # 为每个问题生成SAMPLE_NUM个样本
            batch_prompts = [
                [{"role":"system", "content": sys_set}, 
                {"role": "user", "content": row['problem']}]
                for sys_set in system_settings
            ]
            buffer_msgs.append(batch_prompts)
            buffer_sols.append(row['solution'])
            if (i + 1) % (MAX_NUM_SEQ // len(batch_prompts)) == 0:
                cur_msgs += self._to_buffer(buffer_msgs, buffer_sols)
                buffer_msgs.clear()
                buffer_sols.clear()
                self.log_info(cur_msgs)
            i += 1
        self.cur_data_idx += i
        self._write_buffer(cur_msgs[:INT_NUM * REP_NUM])
        print('结束采样下标：', self.cur_data_idx)

    def _generate_batch(self, prompts):
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
        outputs = self.llm.generate(
            formatted_prompts,
            sampling_params=sampling_params,
        )
        torch.cuda.empty_cache()
        return [
            prompt + [{"role": "assistant", "content": output.outputs[0].text}]
            for prompt, output in zip(prompts, outputs)
        ]

    def _write_buffer(self, data):
        random.shuffle(data) # 打乱数据
        
        self.log_info()
        self.clear_info()

        with open(buffer_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已写入{len(data)}条样本到缓冲区")
    
    def train_model(self):
        print("\n--- 开始训练阶段 ---")
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        if GPU_NUM == 1:
            # for single gpu use deepspeed_zero2
            os.system(f'CUDA_VISIBLE_DEVICES={GPU} accelerate launch --config_file "{current_dir}/reason_llm/deepspeed_zero2.yaml" "{current_dir}/reason_llm/grpo_trainer.py"')
        else:
            # for multi gpu use deepspeed_zero3
            os.system(f'CUDA_VISIBLE_DEVICES={GPU} accelerate launch --config_file "{current_dir}/reason_llm/deepspeed_zero3.yaml" "{current_dir}/reason_llm/grpo_trainer.py"')

    def test_model(self):
        print("\n--- 开始测试阶段 ---")
        buffer_msgs = []
        buffer_sols = []

        data = []
        with open(test_data_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        self.clear_info()
        for i in range(len(data)):
            row = data[i]
            # 为每个问题生成SAMPLE_NUM个样本
            batch_prompts = [
                [{"role":"system", "content": sys_set},
                {"role": "user", "content": row['question']}]
                for sys_set in system_settings
            ]
            buffer_msgs.append(batch_prompts)
            buffer_sols.append('\\boxed{' + row['answer'] + '}')

            if (i + 1) % (MAX_NUM_SEQ // len(batch_prompts)) == 0:
                self._to_buffer(buffer_msgs, buffer_sols)
                buffer_msgs.clear()
                buffer_sols.clear()
                self.log_info()

        if len(buffer_msgs) > 0:
            self._to_buffer(buffer_msgs, buffer_sols)

        self.log_info()
        self.clear_info()

    def run_cycle(self):
        self.generate_samples()
        self.train_model() # lora：base model_dir lora lora
        self.load_model() # merge : base + lora
        self.test_model()
