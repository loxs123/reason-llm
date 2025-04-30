import os
import re

# Get the parent directory of the current file
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths for model, logs, and data buffer file
model_dir = os.path.join(current_dir, "model")  # Directory to store trained models
log_dir = os.path.join(current_dir, "log")  # Directory to store training logs
data_dir = os.path.join(current_dir, "data")  # File to store data buffer
config_file = os.path.join(current_dir, 'reason_llm', 'config.py')

# Training-related parameters
per_device_train_batch_size = 4  # Training batch size per device; a larger batch size improves stability
gradient_accumulation_steps = 32  # Number of gradient accumulation steps to simulate a larger batch size and reduce memory usage

# GPU-related parameters
GPU = "0,1,2,3"  # GPU device IDs to train model
GPU_NUM = len(GPU.split(","))  # Number of available GPUs

GENERATE_BACKEND = 'vllm' # or vllm[recommend]/lmdeploy/sglang
GENERATE_GPU_CONFIG = ['0','1','2','3']  # vLLM resource allocation, where each string represents a GPU allocation
GENERATE_PER_WORKER_GPU = len(GENERATE_GPU_CONFIG[0].split(','))  # Number of GPUs allocated per vLLM task

# Ensure all vLLM configurations have the same number of assigned GPUs
assert len(set([len(v.split(',')) for v in GENERATE_GPU_CONFIG])) == 1, "every vllm same"

# Model-related parameters
MAX_MODEL_LEN = 4096  # Maximum model input length in tokens
TRAIN_MAX_GENERATE_LEN = 3000
EVAL_MAX_GENERATE_LEN = 3000
MAX_NUM_SEQ = 48 * len(GENERATE_GPU_CONFIG)  # The number of sequences processed simultaneously per vLLM worker
INT_NUM = 1024  # The number of sequences per training iteration
ITER_NUM = 200  # The number of iter
TEST_FREQ = 5  # 
assert INT_NUM % (per_device_train_batch_size * gradient_accumulation_steps * GPU_NUM) == 0

# Training hyperparameters
FORMAT_WEIGHT = 0.0    # Weight for format matching
ACCURACY_WEIGHT = 1.0  # Weight for accuracy

BETA = 0.0  # KL divergence loss weight for reference model (used for loss control)
EPSILON_LOW = 0.2  # PPO algorithm clip ratio, controlling the update magnitude of the policy
EPSILON_HIGH = 0.2  # PPO algorithm clip ratio, controlling the update magnitude of the policy
LR = 3e-6  # Learning rate
KL_ESTIMATOR = 'k2' # 
USE_TOKEN_LEVEL_ADV = 0
TOKEN_LEVEL_BETA = 0.2
USE_GPG = True
USE_DYNAMIC_BATCH = True
SCALE_ADV_WITH_STD = False  # Whether to scale the reward

# Generation-related parameters
NUM_GENERATIONS = 8 # 
SYS_SET = 'Please reason step by step, and put your final answer within \\boxed{}.'

# Ensure MAX_NUM_SEQ is divisible by NUM_GENERATIONS
assert MAX_NUM_SEQ % NUM_GENERATIONS == 0

# Training and testing datasets
TRAIN_DATASET = "/root/lanyun-tmp/reason-llm/data/math_12k"  # Training dataset
TEST_DATASETS = ["/root/lanyun-tmp/reason-llm/data/aime",  # 30
                 '/root/lanyun-tmp/reason-llm/data/minerva', # 272
                 '/root/lanyun-tmp/reason-llm/data/olympiad_bench', # 675
                 '/root/lanyun-tmp/reason-llm/data/amc', # 45
                 '/root/lanyun-tmp/reason-llm/data/math', # 500
                ]

START_TRAIN_IDX = 0 # The train_idx from the last training process.

# Predefined assistant role token
ASSISTANT_TOKEN = 'assistant' # 

# 需要根据数据集格式编写build_msgs函数和build_sol函数
def build_msgs(row, dataset, num_generations):
    question = row['problem']
    # {"role":"system", "content": SYS_SET}, 
    msgs = [
        [{"role":"system", "content": SYS_SET}, 
         {"role": "user", "content": question}]
        for _ in range(num_generations)
    ]

    return msgs

def build_sol(row, dataset):
    # row中所有的key已经变为小写
    return row['answer']
    # return '\\boxed{'+ row['answer'] + '}'
