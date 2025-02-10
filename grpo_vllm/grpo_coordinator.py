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

from .grpo_reward_fn import group_reward_fn
from .utils import apply_lora, ThinkCountLogitsProcessor

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# model_dir = os.path.join(current_dir, "model")
model_dir = '/root/shared-storage/DeepSeek-R1-Distill-Qwen-14B'
log_dir = os.path.join(current_dir, "log")
buffer_file = os.path.join(current_dir, "data", "buffer.json")
data_file = os.path.join(current_dir, "data", "train.csv")
test_data_file = os.path.join(current_dir, "data", "test.csv")
MAX_MODEL_LEN = 12192
SAMPLE_NUM = 8
MAX_NUM_SEQ = 32
INT_NUM = 128

random.seed((time.time() // 10) % 124291)

GPU_NUM = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))

assert MAX_NUM_SEQ % SAMPLE_NUM == 0

system_settings = [
    r"You are a the most powerful math expert. Please solve the problems with deep resoning. You are careful and always recheck your conduction. You will never give answer directly until you have enough confidence. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000.",
    r"You are a reliable math expert, known for accurate and well-reasoned solutions. Work through the problem methodically, rechecking your logic and calculations. Only when you're certain of your answer, provide it in \boxed{}, and take it modulo 1000.",
    r"Act as a conscientious math solver. Approach the problem with care, ensuring each part is understood and correctly addressed. Once you're confident in your solution, present it in \boxed{}, and don't forget to compute it modulo 1000.",
    r"You are a deep-thinking math expert. Engage with the problem thoughtfully, exploring underlying principles. After thorough analysis and confirmation, box your answer in \boxed{}, and adjust it to modulo 1000.",
    r"As a precise math assistant, your approach should be step-by-step, with careful consideration of all elements. Ensure you've double-checked your work before finalizing. Present your answer in \boxed{}, and apply modulo 1000.",
    r"You are a thorough math expert. Tackle the problem with meticulous attention to detail, verifying each step along the way. Only when you're fully satisfied with your solution, provide the answer in \boxed{}, and remember to take it modulo 1000.",
    r"Act as a thoughtful math solver. Consider the problem from various perspectives, ensuring comprehensive understanding. Once you've arrived at a solution with high confidence, enclose it in \boxed{}, and adjust it to modulo 1000.",
    r"You are a detailed-oriented math expert. Approach the problem with precision, examining every detail. After multiple validations, present your answer in \boxed{}, and make sure to compute it modulo 1000.",
    r"As an intelligent math assistant, your goal is to solve the problem with deep understanding. Think through each part thoroughly, ensuring no mistakes are made. Once confident, provide the answer in \boxed{}, and apply modulo 1000.",
    r"You are a careful and methodical math solver. Proceed through the problem step-by-step, rechecking your work at each stage. Only when you're absolutely sure of your solution, present it in \boxed{}, and take it modulo 1000.",
    r"Act as a profound math expert. Contemplate the problem extensively, exploring all possible angles. Once you've reached a conclusion with utmost certainty, box your answer in \boxed{}, and adjust it to modulo 1000.",
    r"You are a meticulous problem solver in mathematics. Approach the question with deep thought, ensuring every aspect is considered. After rigorous verification, present your final answer in \boxed{}, and don't forget to apply modulo 1000.",
    r"As a highly skilled math assistant, your approach should be systematic and thorough. Think step-by-step, justifying each move. Only when you're fully confident in your reasoning, provide the answer in \boxed{}, and remember to take it modulo 1000.",
    r"You are a rigorous math expert. Delve into the problem with careful consideration, ensuring no detail is overlooked. After confirming your solution through multiple checks, present the answer in \boxed{}, adjusted to modulo 1000.",
    r"Act as a seasoned mathematician, known for deep analytical skills. Tackle the problem methodically, verifying each logical step. Once you've reached a conclusion with absolute certainty, express the answer enclosed in \boxed{}, and make sure to compute it modulo 1000.",
    r"You are an advanced mathematics solver. Approach the problem with meticulous care, ensuring each step is thoroughly understood before proceeding. Only provide the final answer when you are entirely confident in your solution, and present it within \boxed{}, after applying modulo 1000.",
]

class TrainingSamplingCoordinator:
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.cur_data_idx = 32
        self.acc_ave = []
        self.reward = []
        self.acc_major = []
        self._initialize_components()

    def _initialize_components(self):
        """初始化模型和处理器"""
        self._load_model()
        self.think_processor = ThinkCountLogitsProcessor(
            "<think>", self.tokenizer
        )

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
            gpu_memory_utilization=0.96,
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
                # cur_msgs.append({
                #     "completion": msg,
                #     "advantage": advantage.item(),
                #     "reward": reward.item(),
                #     "label": buffer_labels[j // sample_num],
                # })

                # 只训练好的
                if reward > 0.5:
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
        buffer_msgs = []
        buffer_labels = []
        cur_msgs = []

        data = []
        with open(data_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        random.seed(1323)
        random.shuffle(data)
        
        # for i, row in enumerate(data):
        i = 0
        while len(cur_msgs) < INT_NUM:

            row = data[(self.cur_data_idx + i) % len(data)]
            
            # 为每个问题生成SAMPLE_NUM个样本
            batch_prompts = [
                [{"role": "user", "content": row['question'] + '\nIf the final answer is a number larger than 1000, take modulo 1000.'}]
                for _ in range(SAMPLE_NUM)
            ]

            buffer_msgs.append(batch_prompts)
            buffer_labels.append(row['answer'])

            # self.cur_data_idx += 1
            if (i + 1) % (MAX_NUM_SEQ // SAMPLE_NUM) == 0:
                cur_msgs += self._to_buffer(buffer_msgs, buffer_labels)
                buffer_msgs.clear()
                buffer_labels.clear()
                print(f'平均正确率：{sum(self.acc_ave) / len(self.acc_ave)}，多数投票正确率：{sum(self.acc_major) / len(self.acc_major)}，平均奖励：{sum(self.reward) / len(self.reward)}，收集轨迹数量：{len(cur_msgs)}')
            
            i += 1
                
        self.cur_data_idx += i
        self._write_buffer(cur_msgs)

    def _generate_batch(self, prompts):
        """执行批量生成"""
        sampling_params = SamplingParams(
            temperature=0.6,
            min_p=0.01,
            max_tokens=MAX_MODEL_LEN,
            skip_special_tokens=True,
            logits_processors=[self.think_processor],
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
        os.system(f'CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file "{current_dir}/grpo_vllm/deepspeed_zero3.yaml" "{current_dir}/grpo_vllm/grpo_trainer.py"')


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
        data = data[:]
        self.acc_ave.clear()
        self.acc_major.clear()
        self.reward.clear()
        
        for i in range(len(data)):

            row = data[i]
            
            # 为每个问题生成SAMPLE_NUM个样本
            batch_prompts = [
                [{"role": "user", "content": row['question'] + '\n' + sys_set}] for sys_set in system_settings
            ]

            buffer_msgs.append(batch_prompts)
            buffer_labels.append(row['answer'])

            # self.cur_data_idx += 1
            if (i + 1) % (MAX_NUM_SEQ // 16) == 0:
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
