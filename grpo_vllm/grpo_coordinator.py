import os
import time
import sys
import datetime
import json
import csv
import copy
import gc
import numpy as np
import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from grpo_vllm import GRPOTrainer, GRPOConfig, GRPODataset, group_reward_fn
from .utils import apply_lora, ThinkCountLogitsProcessor

model_dir = os.path.join(current_dir, "model")
log_dir = os.path.join(current_dir, "log")
buffer_file = os.path.join(current_dir, "data", "buffer.json")
data_file = os.path.join(current_dir, "data", "train.csv")
MAX_MODEL_LEN = 8192
SAMPLE_NUM = 8
MAX_NUM_SEQ = 32
INT_NUM = 512

GPU_NUM = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))

assert MAX_NUM_SEQ % SAMPLE_NUM == 0

class TrainingSamplingCoordinator:
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.cur_data_idx = 0
        self._initialize_components()

    def _initialize_components(self):
        """初始化模型和处理器"""
        self._load_model()
        self.think_processor = ThinkCountLogitsProcessor(
            "<think>", self.tokenizer
        )

    def _load_model(self):
        """加载或重新加载模型"""
        if self.llm is not None:
            del self.llm
            gc.collect()
            torch.cuda.empty_cache()

        self.llm = self._create_llm_instance()
        self.tokenizer = copy.deepcopy(self.llm.get_tokenizer())

    def _create_llm_instance(self):
        """创建LLM实例"""

        apply_lora()
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
        completed_batch = self._generate_batch(batch_prompts)
        cur_msgs = []
        for j in range(0, len(completed_batch), SAMPLE_NUM):

            # 处理结果
            rewards = group_reward_fn(
                prompts=None,
                completions=completed_batch[j:j+SAMPLE_NUM],
                label=buffer_labels[j // SAMPLE_NUM],
            )
            rewards = np.array(rewards)
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

            # 保存到当前批次
            for msg, advantage in zip(completed_batch[j:j+SAMPLE_NUM], advantages):
                cur_msgs.append({
                    "completion": msg,
                    "advantage": advantage.item(),
                    "label": buffer_labels[j // SAMPLE_NUM],
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
    
        # for i, row in enumerate(data):
        for i in range(INT_NUM):

            row = data[(self.cur_data_idx + i) % len(data)]
            # 构造基础提示
            
            sys_set = ("You are a the most powerful math expert. "
                    "Please solve the problems with deep resoning. "
                    "You are careful and always recheck your conduction. "
                    "You will never give answer directly until you have enough confidence. "
                    "You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000.")
            
            # 为每个问题生成SAMPLE_NUM个样本
            batch_prompts = [
                [{"role":"system", "content":sys_set},
                    {"role": "user", "content": row['question']}]
                for _ in range(SAMPLE_NUM)
            ]

            buffer_msgs.append(batch_prompts)
            buffer_labels.append(row['answer'])

            self.cur_data_idx += 1
            if (i + 1) % (MAX_NUM_SEQ // SAMPLE_NUM) == 0:
                cur_msgs += self._to_buffer(buffer_msgs, buffer_labels)
                buffer_msgs.clear()
                buffer_labels.clear()

            if len(cur_msgs) >= INT_NUM:
                self._write_buffer(cur_msgs)
                break

    def _generate_batch(self, prompts):
        """执行批量生成"""
        sampling_params = SamplingParams(
            temperature=1.0,
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

        # 构造完整对话记录
        return [
            prompt + [{"role": "assistant", "content": output.outputs[0].text}]
            for prompt, output in zip(prompts, outputs)
        ]

    def _write_buffer(self, data):
        """写入缓冲区文件"""
        with open(buffer_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已写入{len(data)}条样本到缓冲区")

    def train_model(self):
        """执行模型训练"""
        print("\n--- 开始训练阶段 ---")
        
        # 等待数据就绪
        while not os.path.exists(buffer_file):
            print("等待采样数据...")
            time.sleep(10)

        # 加载数据集
        train_dataset = GRPODataset(buffer_file, data_file, SAMPLE_NUM)
        
        # 初始化模型
        if os.path.exists(model_dir, 'lora'):
            model = AutoModelForCausalLM.from_pretrained(model_dir)
            print(f"Loading the LoRA adapter from {os.path.exists(model_dir, 'lora')}")
            lora_model = PeftModel.from_pretrained(
                model,
                os.path.exists(model_dir, 'lora'),
                torch_dtype=torch.float16,
            )
            model = lora_model.merge_and_unload()
        elif os.path.exists(os.path.join(model_dir, 'merge')):
            print(f"Loading model from {os.path.exists(model_dir, 'merge')}")
            model = AutoModelForCausalLM.from_pretrained(os.path.join(model_dir, 'merge'))
        else:
            print(f"Loading model from {model_dir}")
            model = AutoModelForCausalLM.from_pretrained(model_dir)

        model.gradient_checkpointing_enable()

        # 配置LoRA
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_alpha=64,
            lora_dropout=0.1,
            bias="none",
        )

        # 配置训练参数
        training_args = GRPOConfig(
            output_dir=os.path.join(model_dir, 'lora'), # lora
            # output_dir=os.path.join(model_dir, 'merge'), # full
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            save_strategy="epoch",
            logging_dir=os.path.join(log_dir, f"experiment_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"),
            report_to="tensorboard",
            overwrite_output_dir=True,
            save_only_model=True,
            weight_decay=0.01,
            bf16=True,
            logging_steps=5,
            log_level="info",
        )

        # 执行训练
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config,
        )
        trainer.train()
        
        # 清理训练资源
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    def run_cycle(self):
        """执行完整的训练-采样周期"""
        try:
            # 1. 采样阶段
            self.generate_samples()
            # 2. 训练阶段
            self.train_model()
            # 3. 更新模型
            self._load_model()
        except Exception as e:
            print(f"运行时错误: {str(e)}")
            self._load_model()  # 出错时重新加载模型

    