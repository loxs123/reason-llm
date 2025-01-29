import os
import time
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

from .grpo_trainer import GRPOTrainer
from .grpo_config import GRPOConfig
from .grpo_dataset import GRPODataset
from .grpo_reward_fn import reward_fn

if __name__ == '__main__':

    current_dir =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    last_time = time.time()
    model_dir = os.path.join(current_dir, 'model')
    log_dir = os.path.join(current_dir, 'log')
    buffer_file = os.path.join(current_dir, 'data', 'buffer.json')

    while not os.path.exists(buffer_file):
        time.sleep(10)
    
    print('find buffer.json, start load data...')
    train_dataset = GRPODataset(buffer_file)

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 获取精确到毫秒的当前时间
    formatted_now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    experiment_name = f"experiment_{formatted_now}"

    # 设置训练参数
    training_args = GRPOConfig(
        output_dir=model_dir, 
        num_train_epochs=10, # 
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{log_dir}/{experiment_name}",
        report_to="tensorboard",               # 启用 TensorBoard
        overwrite_output_dir=True, 
        save_only_model=True,
        weight_decay=0.01, # L2正则化
    )

    trainer = GRPOTrainer(model, [reward_fn], training_args, train_dataset)
    trainer.train()
