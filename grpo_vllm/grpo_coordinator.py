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

from .grpo_reward_fn import group_reward_fn
from .utils import apply_lora, ThinkCountLogitsProcessor

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(current_dir, "autodl-tmp/model")
# model_dir = '/root/shared-storage/DeepSeek-R1-Distill-Qwen-14B'
log_dir = os.path.join(current_dir, "log")
buffer_file = os.path.join(current_dir, "data", "buffer.json")
data_file = os.path.join(current_dir, "data", "train.csv")
test_data_file = os.path.join(current_dir, "data", "test.csv")
system_setting_file = os.path.join(current_dir, "grpo_vllm/system_setting.txt")
MAX_MODEL_LEN = 2048
MAX_NUM_SEQ = 32
INT_NUM = 128

random.seed((time.time() // 10) % 124291)

GPU_NUM = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))

with open(system_setting_file) as f:
    d = f.read()
    system_settings = re.findall(r'```text\n(.*?)\n```', d, re.S)

assert MAX_NUM_SEQ % len(system_settings) == 0

class TrainingSamplingCoordinator:
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.cur_data_idx = 0
        self.acc_ave = []
        self.reward = []
        self.acc_major = []
        self._initialize_components()

    def _initialize_components(self):
        """初始化模型和处理器"""
        self._load_model()
        self._init_data()
        
    def _init_data(self):
        
        data = []
        with open(data_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        random.seed(1323)
        random.shuffle(data)

        self.data = data

        print('数据集总量：', len(self.data))


    def _load_model(self):
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

    def _to_buffer(self, buffer_msgs, buffer_labels):
        
        # INT_NUM
        batch_prompts = [msg for msgs in buffer_msgs for msg in msgs]
        sample_num = len(buffer_msgs[0])
        completed_batch = self._generate_batch(batch_prompts)
        cur_msgs = []
        for j in range(0, len(completed_batch), sample_num):

            # 处理结果
            rewards, correct, acc_rate = group_reward_fn(
                prompts=None,
                completions=completed_batch[j : j+sample_num],
                label=buffer_labels[j // sample_num],
            )
            self.reward.extend(rewards)
            self.acc_ave.append(acc_rate)
            self.acc_major.append(correct)

            rewards = np.array(rewards)

            # if rewards.std() < 0.05:
            #     continue
                
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

            # 保存到当前批次
            for msg, advantage,reward in zip(completed_batch[j:j+sample_num], advantages, rewards):
                cur_msgs.append({
                    "completion": msg,
                    "advantage": advantage.item(),
                    "reward": reward.item(),
                    "label": buffer_labels[j // sample_num],
                })

        return cur_msgs

    def generate_samples(self):
        """生成样本并保存到缓冲区"""
        print("\n--- 开始采样阶段 ---")

        print('开始采样下标：', self.cur_data_idx)

        data_size = len(self.data)

        buffer_msgs = []
        buffer_labels = []
        cur_msgs = []
        
        # for i, row in enumerate(data):
        i = 0
        while len(cur_msgs) < INT_NUM:
            
            row = self.data[(self.cur_data_idx + i) % data_size]
            
            # 为每个问题生成SAMPLE_NUM个样本
            batch_prompts = [
                [{"role":"system", "content": sys_set}, 
                {"role": "user", "content": row['question']}]
                for sys_set in system_settings
            ]

            buffer_msgs.append(batch_prompts)
            buffer_labels.append(row['answer'])

            # self.cur_data_idx += 1
            if (i + 1) % (MAX_NUM_SEQ // len(batch_prompts)) == 0:
                cur_msgs += self._to_buffer(buffer_msgs, buffer_labels)
                buffer_msgs.clear()
                buffer_labels.clear()
                print(f'平均正确率：{sum(self.acc_ave) / len(self.acc_ave)}，多数投票正确率：{sum(self.acc_major) / len(self.acc_major)}，平均奖励：{sum(self.reward) / len(self.reward)}，收集轨迹数量：{len(cur_msgs)}')
            
            i += 1

        self.cur_data_idx += i
        self._write_buffer(cur_msgs)

        print('结束采样下标：', self.cur_data_idx)

    def _generate_batch(self, prompts):
        """执行批量生成"""
        sampling_params = SamplingParams(
            temperature=1.0,
            min_p=0.01,
            max_tokens=MAX_MODEL_LEN,
            skip_special_tokens=True,
        )

        # 将提示转换为模型输入格式
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]

        # 执行生成
        outputs = self.llm.generate(
            formatted_prompts,
            sampling_params=sampling_params,
        )
        torch.cuda.empty_cache()

        # 构造完整对话记录
        return [
            prompt + [{"role": "assistant", "content": output.outputs[0].text}]
            for prompt, output in zip(prompts, outputs)
        ]

    def _write_buffer(self, data):

        if len(data) == 0:
            print('没有样本写入到缓冲区')
            return
        
        print(f'平均正确率：{sum(self.acc_ave) / len(self.acc_ave)}，多数投票正确率：{sum(self.acc_major) / len(self.acc_major)}，平均奖励：{sum(self.reward) / len(self.reward)}')
        self.acc_ave.clear()
        self.acc_major.clear()
        self.reward.clear()

        """写入缓冲区文件"""
        with open(buffer_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已写入{len(data)}条样本到缓冲区")

    def train_model(self):
        """执行模型训练"""
        print("\n--- 开始训练阶段 ---")

        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        # for single gpu use deepspeed_zero2
        os.system(f'CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file "{current_dir}/grpo_vllm/deepspeed_zero2.yaml" "{current_dir}/grpo_vllm/grpo_trainer.py"')
        # for multi gpu use deepspeed_zero3
        # os.system(f'CUDA_VISIBLE_DEVICES=0,1,2,4 accelerate launch --config_file "{current_dir}/grpo_vllm/deepspeed_zero3.yaml" "{current_dir}/grpo_vllm/grpo_trainer.py"')

    def test_model(self):
        print("\n--- 开始测试阶段 ---")
        buffer_msgs = []
        buffer_labels = []
        cur_msgs = []

        data = []
        with open(test_data_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        self.acc_ave.clear()
        self.acc_major.clear()
        self.reward.clear()
        
        for i in range(len(data)):

            row = data[i]
            
            # 为每个问题生成SAMPLE_NUM个样本
            batch_prompts = [
                [{"role":"system", "content": sys_set},
                {"role": "user", "content": row['question']}]
                for sys_set in system_settings
            ]
            buffer_msgs.append(batch_prompts)
            buffer_labels.append(row['answer'])

            # self.cur_data_idx += 1
            if (i + 1) % (MAX_NUM_SEQ // len(batch_prompts)) == 0:
                cur_msgs += self._to_buffer(buffer_msgs, buffer_labels)
                buffer_msgs.clear()
                buffer_labels.clear()
                print(f'平均正确率：{sum(self.acc_ave) / len(self.acc_ave)}，多数投票正确率：{sum(self.acc_major) / len(self.acc_major)}，平均奖励：{sum(self.reward) / len(self.reward)}')

        if len(buffer_msgs) > 0:
            cur_msgs += self._to_buffer(buffer_msgs, buffer_labels)

        print(f'平均正确率：{sum(self.acc_ave) / len(self.acc_ave)}，多数投票正确率：{sum(self.acc_major) / len(self.acc_major)}，平均奖励：{sum(self.reward) / len(self.reward)}')
        
        self.acc_ave.clear()
        self.acc_major.clear()
        self.reward.clear()

    def run_cycle(self):
        """执行完整的训练-采样周期"""
        # self._load_model() 0 base
        # 1. 采样阶段
        self.generate_samples()
        # 2. 训练阶段
        self.train_model() # lora：base model_dir lora lora
        # 3. 更新模型
        self._load_model() # merge : base + lora
        # 4. 测试模型
        self.test_model()

