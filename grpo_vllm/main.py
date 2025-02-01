import os
import time
import sys
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

current_dir =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from peft import LoraConfig

from grpo_vllm import GRPOTrainer
from grpo_vllm import GRPOConfig
from grpo_vllm import GRPODataset

if __name__ == '__main__':

    last_time = time.time()
    model_dir = os.path.join(current_dir, 'model')
    log_dir = os.path.join(current_dir, 'log')
    buffer_file = os.path.join(current_dir, 'data', 'buffer.json')
    train_file = os.path.join(current_dir, 'data', 'train.csv')
    
    print('waiting buffer...')
    while not os.path.exists(buffer_file):
        time.sleep(10)
    
    print('find buffer.json, start load data...')
    train_dataset = GRPODataset(buffer_file, train_file, sample_num = 8)

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 获取精确到毫秒的当前时间
    formatted_now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    experiment_name = f"experiment_{formatted_now}"

    # 设置训练参数
    training_args = GRPOConfig(
        output_dir=model_dir, 
        num_train_epochs=2, # 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_strategy="no",
        logging_dir=f"{log_dir}/{experiment_name}",
        report_to="tensorboard",               # 启用 TensorBoard
        overwrite_output_dir=True, 
        save_only_model=True,
        weight_decay=0.01, # L2正则化
        bf16=True,  # 启用 bf16
        # gradient_checkpointing=True,  # 启用梯度检查点 降低显存
    )

    # 使用peft
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=32,  
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  
        lora_alpha=64,  
        lora_dropout=0.1,  
        bias="none",  
        use_rslora=False,  
        modules_to_save=None  
    )

    trainer = GRPOTrainer(model, training_args, train_dataset, peft_config=peft_config)
    trainer.train()

    # # 全量
    # trainer = GRPOTrainer(model, [reward_fn], training_args, train_dataset)
    # trainer.train()
