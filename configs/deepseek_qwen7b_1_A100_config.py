import os
import re

# Get the parent directory of the current file
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths for model, logs, and data buffer file
model_dir = os.path.join(current_dir, "model")  # Directory to store trained models
log_dir = os.path.join(current_dir, "log")  # Directory to store training logs
buffer_file = os.path.join(current_dir, "data", "buffer.json")  # File to store data buffer

# Training-related parameters
per_device_train_batch_size = 4  # Training batch size per device; a larger batch size improves stability
gradient_accumulation_steps = 64  # Number of gradient accumulation steps to simulate a larger batch size and reduce memory usage

# GPU-related parameters
GPU = "0"  # GPU device IDs to train model
GPU_NUM = len(GPU.split(","))  # Number of available GPUs

VLLM_CONFIG = ['0']  # vLLM resource allocation, where each string represents a GPU allocation
PER_VLLM_GPU = len(VLLM_CONFIG[0].split(','))  # Number of GPUs allocated per vLLM task

# Ensure all vLLM configurations have the same number of assigned GPUs
assert len(set([len(v.split(',')) for v in VLLM_CONFIG])) == 1, "every vllm same"

# Model-related parameters
MAX_MODEL_LEN = 8096  # Maximum model input length in tokens
MAX_NUM_SEQ = 32 * len(VLLM_CONFIG)  # The number of sequences processed simultaneously per vLLM worker
INT_NUM = 1024  # The number of sequences per training iteration
REP_NUM = 1  # The number of times each sequence is repeated in a training iteration

assert (INT_NUM * REP_NUM) % (per_device_train_batch_size * gradient_accumulation_steps * GPU_NUM) == 0

# Training hyperparameters
FORMAT_WEIGHT = 0.0  # Weight for format matching
ACCURACY_WEIGHT = 1.0  # Weight for accuracy

BETA = 0.0  # KL divergence loss weight for reference model (used for loss control)
EPSILON_LOW = 0.2  # PPO algorithm clip ratio, controlling the update magnitude of the policy
EPSILON_HIGH = 0.2  # PPO algorithm clip ratio, controlling the update magnitude of the policy
LR = 3e-6  # Learning rate
KL_ESTIMATOR = 'k3' # 
USE_TOKEN_LEVEL_ADV = 0
TOKEN_LEVEL_BETA = 0.2

# Generation-related parameters
NUM_GENERATIONS = 8 # 
SYS_SET = ("You are a seasoned science expert, recognized for your thorough analysis and precise calculations."
          "Work through the given problem step by step, ensuring every detail is carefully considered. "
          "Present your solution inside <think></think> tags, with the final result enclosed in \\boxed{}.")

SYS_SETS = [SYS_SET for _ in range(NUM_GENERATIONS)]  # Duplicate system instruction template to ensure consistency across generation tasks

# Ensure MAX_NUM_SEQ is divisible by NUM_GENERATIONS
assert MAX_NUM_SEQ % NUM_GENERATIONS == 0

# Training and testing datasets
TRAIN_DATASET = "di-zhang-fdu/AIME_1983_2024"  # Training dataset
TEST_DATASET = "BBexist/AIME25"  # Testing dataset
START_TRAIN_IDX = 0 # The train_idx from the last training process.

# Predefined assistant role token
ASSISTANT_TOKEN = '<｜Assistant｜>' # 
# https://zhuanlan.zhihu.com/p/21465667399
# 需要调整一下 tokenizer_config.json 中的字段，参照博客内容

# 需要根据数据集格式编写build_msgs函数和build_sol函数
def build_msgs(row, mode='train'):
    # row中所有的key已经变为小写
    assert mode in {"train", "test"}
    if mode == 'train':
        question = row['question']
        msgs = [
            [{"role":"system", "content": sys_set}, 
            {"role": "user", "content": question}]
            for sys_set in SYS_SETS
        ]
    else:
        question = row['problem']
        msgs = [
            [{"role":"system", "content": sys_set}, 
            {"role": "user", "content": question}]
            for sys_set in SYS_SETS
        ]

    return msgs

def build_sol(row, mode='train'):
    # row中所有的key已经变为小写
    assert mode in {"train", "test"}
    if mode == 'train':
        return '\\boxed{' + row['answer'] + '}'
    else:
        return '\\boxed{' + row['answer'] + '}'