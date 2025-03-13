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
import glob
import shutil
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data as th_data
import json
import time
import random
import transformers
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.models import prepare_deepspeed
from trl.data_utils import maybe_apply_chat_template


from peft import LoraConfig, PeftModel

from reason_llm.utils import *
from reason_llm.config import *

class GRPODataset(th_data.Dataset):
    def __init__(self, filename):
        super(GRPODataset, self).__init__()
        with open(filename) as f:
            data = json.load(f)
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item

@dataclass
class GRPOConfig(TrainingArguments):
    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=LR,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=BETA, 
        metadata={"help": "KL coefficient."},
    )

    epsilon: float = field(
        default=EPSILON,
        metadata={"help": "KL coefficient."},
    )

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GRPOTrainer(Trainer):

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # Reference model
        if args.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        

        if os.path.exists(os.path.join(model_dir, 'merge')):
            pre_model_id = os.path.join(model_dir, 'merge')
        else:
            pre_model_id = model_dir
            
        if is_deepspeed_zero3_enabled():
            self.ref_model2 = AutoModelForCausalLM.from_pretrained(pre_model_id)
        else:
            self.ref_model2 = AutoModelForCausalLM.from_pretrained(pre_model_id)
            for param in self.ref_model2.parameters():
                param.requires_grad = False
            self.ref_model2.eval()

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        if processing_class.pad_token_id is None:
            processing_class.pad_token_id = processing_class.eos_token_id
        self.peft_config = peft_config

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        self.beta = args.beta
        self.epsilon = args.epsilon

        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if self.is_deepspeed_enabled:
            self.ref_model2 = prepare_deepspeed(self.ref_model2, self.accelerator)
        else:
            self.ref_model2 = self.accelerator.prepare_model(self.ref_model2, evaluation_mode=True)


    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    def _logp(self, logps, seq_len, device):
        
        for i in range(len(logps)):
            if len(logps[i]) > seq_len:
                logps[i] = logps[i][: seq_len]
            else:
                logps[i] = logps[i] + [0.0] * (seq_len - len(logps[i]))
        return torch.tensor(logps, dtype=torch.float32, device=device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """inputs = [
                {'completion': [ {'role': 'user', 'content': 'question'}, {role': 'assistant', 'content': 'answer'}], 'advantage': 0.8, 'label': '12'},
                ...   
            ]
        """

        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        prompts_text = [maybe_apply_chat_template({'messages': example['completion'] }, self.processing_class)["text"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
        )['input_ids']

        seq_len = prompt_inputs.size(1) - 1

        assistant_id = self.processing_class(ASSISTANT_TOKEN, add_special_tokens=False)['input_ids'][0]

        pad_id = self.processing_class.pad_token_id
        prefix_mask = create_prefix_mask(prompt_inputs, assistant_id)
        suffix_mask = create_suffix_mask(prompt_inputs, pad_id)

        mask = prefix_mask * suffix_mask
        
        prompt_completion_ids = super()._prepare_inputs(prompt_inputs)
        mask = super()._prepare_inputs(mask)[:, 1: ] # [bs, sl]
        
        device = self.accelerator.device

        per_token_logps = get_per_token_logps(model, prompt_completion_ids)

        with torch.inference_mode():
            old_per_token_logps = get_per_token_logps(self.ref_model2, prompt_completion_ids)
        # old_per_token_logps = [example['old_per_token_logps'] for example in inputs]
        # old_per_token_logps = self._logp(old_per_token_logps, seq_len, device)

        advantages = [example['advantage'] for example in inputs]
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device) # batch_size

        # x - x.detach() allows for preserving gradients from x
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta > 0.0:
            # [2 - last] logps
            # ref_per_token_logps = [example['ref_per_token_logps'] for example in inputs]
            # ref_per_token_logps = self._logp(ref_per_token_logps, seq_len, device)

            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids)
            
            # Compute the KL divergence between the model and the reference model
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)).mean()
        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-4)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * mask).sum() / mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

if __name__ == '__main__':

    totol = INT_NUM * REP_NUM # 每次采样1024个，每个样本重复3次

    epoch_steps = totol / per_device_train_batch_size / gradient_accumulation_steps / GPU_NUM

    # 加载数据集
    train_dataset = GRPODataset(buffer_file)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    # 配置训练参数
    training_args = GRPOConfig(
        output_dir=os.path.join(model_dir, 'tmp'), # full
        num_train_epochs=1,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_dir=os.path.join(log_dir, f"experiment_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"),
        report_to="tensorboard",
        overwrite_output_dir=True,
        bf16=True,
        logging_steps=1,
        log_level="info",
        lr_scheduler_type="constant",
    )

    # ############## LORA START ################

    # # 配置LoRA
    # peft_config = LoraConfig(
    #     task_type="CAUSAL_LM",
    #     r=256,
    #     target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     lora_alpha=256,
    #     lora_dropout=0.05,
    #     bias="none",
    # )

    # model = get_peft_model(model, peft_config)
    
    # model.enable_input_require_grads()
    # model.config.use_cache = False  # 梯度检查点与cache不兼容
    # model.base_model.gradient_checkpointing_enable()

    # # 执行训练
    # trainer = GRPOTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     peft_config=peft_config,
    # )

    # checkpoint_paths = glob.glob(os.path.join(model_dir, 'tmp/checkpoint-*'))
    # checkpoint_paths.sort(key = lambda x: int(x.split('-')[-1]))
    
    # if len(checkpoint_paths) == 0:
    #     print('从头开始训练')
    #     trainer.train()
    # else:
    #     checkpoint_path = checkpoint_paths[-1]
    #     if len(checkpoint_paths) >= 2:
    #         print('删除',checkpoint_paths[-2])
    #         shutil.rmtree(checkpoint_paths[-2])

    #     print(f'从{checkpoint_path}开始训练...')
    #     steps = int(checkpoint_path.split('-')[-1])

    #     training_args.num_train_epochs = steps // epoch_steps + 1
    #     trainer.train(checkpoint_path)

    # # for vllm
    # trainer.save_model(os.path.join(model_dir, 'lora'))
    # trainer.tokenizer.save_pretrained(os.path.join(model_dir, 'lora'))

    # ############# LoRA END ###################
    
    # ############## FULL ###################

    model.enable_input_require_grads()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    checkpoint_paths = glob.glob(os.path.join(model_dir, 'tmp/checkpoint-*'))
    checkpoint_paths.sort(key = lambda x: int(x.split('-')[-1]))
    
    if len(checkpoint_paths) == 0:
        print('从头开始训练')
        trainer.train()
    else:
        checkpoint_path = checkpoint_paths[-1]
        if len(checkpoint_paths) >= 2:
            print('删除',checkpoint_paths[-2])
            shutil.rmtree(checkpoint_paths[-2])

        print(f'从{checkpoint_path}开始训练...')
        steps = int(checkpoint_path.split('-')[-1])

        training_args.num_train_epochs = steps // epoch_steps + 1
        trainer.train(checkpoint_path)

    trainer.save_model(os.path.join(model_dir, 'merge'))
    trainer.tokenizer.save_pretrained(os.path.join(model_dir, 'merge'))
