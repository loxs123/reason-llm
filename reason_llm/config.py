import os

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_dir = os.path.join(current_dir, "model")
log_dir = os.path.join(current_dir, "log")
buffer_file = os.path.join(current_dir, "data", "buffer.json")

per_device_train_batch_size = 16 # 大一点的batch_size
gradient_accumulation_steps = 16

test_data_file = os.path.join(current_dir, "data", "test.csv")
system_setting_file = os.path.join(current_dir, "grpo_vllm/system_setting.txt")

MAX_MODEL_LEN = 2048
MAX_NUM_SEQ = 32
INT_NUM = 1024
REP_NUM = 1

GPU = "0"
GPU_NUM = len(GPU.split(","))

FORMAT_WEIGHT = 1.0
ACCURACY_WEIGHT = 1.0

DATASET = "xiaodongguaAIGC/X-R1-7500"