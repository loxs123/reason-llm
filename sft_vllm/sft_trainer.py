# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datetime
import json
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments

from peft import LoraConfig, PeftModel

from sft_vllm.sft_dataset import SFTDataset, DataCollatorForDialog

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_dir = os.path.join(current_dir, "model")
log_dir = os.path.join(current_dir, "log")
buffer_file = os.path.join(current_dir, "data", "buffer.json")
data_file = os.path.join(current_dir, "data", "train.csv")
SAMPLE_NUM = 8

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # 加载数据集
    train_dataset = SFTDataset(buffer_file, tokenizer, sample_num = SAMPLE_NUM)

    data_collator = DataCollatorForDialog(
        tokenizer=tokenizer
    )
    
    # 初始化模型
    # if os.path.exists(os.path.join(model_dir, 'merge')):
    #     print(f"Loading model from {os.path.join(model_dir, 'merge')}")
    #     model = AutoModelForCausalLM.from_pretrained(os.path.join(model_dir, 'merge'))
    # else:
    #     print(f"Loading model from {model_dir}")
    #     model = AutoModelForCausalLM.from_pretrained(model_dir)


    model = AutoModelForCausalLM.from_pretrained(model_dir)
    # 配置训练参数
    training_args = Seq2SeqTrainingArguments(
        # output_dir=os.path.join(model_dir, 'lora'), # lora
        output_dir=os.path.join(model_dir, 'tmp'), # full
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        save_strategy="epoch",
        logging_dir=os.path.join(log_dir, f"experiment_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"),
        report_to="tensorboard",
        overwrite_output_dir=True,
        bf16=True,
        logging_steps=5,
        log_level="info",
        lr_scheduler_type="constant",
        remove_unused_columns=False
    )

    ############## LORA START ################

    # 配置LoRA
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=32,
        target_modules=["q_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, peft_config)

    ############# LoRA END ###################
    
    model.enable_input_require_grads()
    model.config.use_cache = False  # 梯度检查点与cache不兼容
    model.gradient_checkpointing_enable()

    # 执行训练
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        peft_config=peft_config,
    )

    checkpoint_path = 'tmp/checkpoint-4'
    if os.path.exists(os.path.join(model_dir, checkpoint_path ,'trainer_state.json')):
        with open(os.path.join(model_dir, checkpoint_path ,'trainer_state.json')) as f:
            data = json.load(f)
        data['epoch'] = 0.0
        data['global_step'] = 0

        with open(os.path.join(model_dir, checkpoint_path ,'trainer_state.json'), 'w') as f:
            json.dump(data, f, ensure_ascii=False,indent=4)

        trainer.train(os.path.join(model_dir, checkpoint_path))
    else:
        trainer.train()
    
    trainer.save_model(os.path.join(model_dir, 'lora'))
    trainer.tokenizer.save_pretrained(os.path.join(model_dir, 'lora'))
    
    # ############## FULL ###################

    # model.enable_input_require_grads()
    # model.config.use_cache = False
    # model.gradient_checkpointing_enable()

    # trainer = GRPOTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    # )
    # trainer.train()
    # trainer.save_model(os.path.join(model_dir, 'merge'))
    # trainer.tokenizer.save_pretrained(os.path.join(model_dir, 'merge'))
