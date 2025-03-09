import os

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_dir = os.path.join(current_dir, "model")
log_dir = os.path.join(current_dir, "log")
buffer_file = os.path.join(current_dir, "data", "buffer.json")

per_device_train_batch_size = 4
gradient_accumulation_steps = 16

data_file = os.path.join(current_dir, "data", "train.csv")
test_data_file = os.path.join(current_dir, "data", "test.csv")
system_setting_file = os.path.join(current_dir, "grpo_vllm/system_setting.txt")
MAX_MODEL_LEN = 2048
MAX_NUM_SEQ = 32
INT_NUM = 1024
REP_NUM = 1
GPU_NUM = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))

