import os
import re

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_dir = os.path.join(current_dir, "model")
log_dir = os.path.join(current_dir, "log")
buffer_file = os.path.join(current_dir, "data", "buffer.json")

per_device_train_batch_size = 16 # 大一点的batch_size
gradient_accumulation_steps = 16 

MAX_MODEL_LEN = 2048
MAX_NUM_SEQ = 32
INT_NUM = 1024
REP_NUM = 1

GPU = "0"
GPU_NUM = len(GPU.split(","))

VLLM_CONFIG = ['0']  # vLLM resource allocation, where each string represents a GPU allocation
PER_VLLM_GPU = len(VLLM_CONFIG[0].split(','))  # Number of GPUs allocated per vLLM task

# Ensure all vLLM configurations have the same number of assigned GPUs
assert len(set([len(v.split(',')) for v in VLLM_CONFIG])) == 1, "every vllm same"

FORMAT_WEIGHT = 1.0
ACCURACY_WEIGHT = 1.0

BETA = 0.0 # ref model kl loss
EPSILON = 0.2 # ppo clip radio
LR = 3e-6

NUM_GENERATIONS = 8
SYS_SET = ("A conversation between User and Assistant. The user asks a question, "
            "and the Assistant solves it. The assistant first thinks about the "
            "reasoning process in the mind and then provides the user with the answer. "
            "The reasoning process and answer are enclosed within <think> </think> "
            "and <answer> </answer> tags, respectively, i.e., <think> reasoning process"
            " here </think><answer> answer here </answer>")

SYS_SETS = [SYS_SET for _ in range(NUM_GENERATIONS)]

assert MAX_NUM_SEQ % NUM_GENERATIONS == 0

TRAIN_DATASET = "xiaodongguaAIGC/X-R1-7500"
TEST_DATASET = "HuggingFaceH4/aime_2024"

ASSISTANT_TOKEN = 'assistant'
