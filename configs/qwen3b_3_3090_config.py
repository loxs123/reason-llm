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
gradient_accumulation_steps = 16  # Number of gradient accumulation steps to simulate a larger batch size and reduce memory usage

# Model-related parameters
MAX_MODEL_LEN = 2048  # Maximum model input length in tokens
MAX_NUM_SEQ = 32 * 3  # The number of sequences processed simultaneously per vLLM worker
INT_NUM = 12 * 16 * 4  # The number of sequences per training iteration
REP_NUM = 1  # The number of times each sequence is repeated in a training iteration

# GPU-related parameters
GPU = "0,1,2"  # GPU device IDs to train model
GPU_NUM = len(GPU.split(","))  # Number of available GPUs

VLLM_CONFIG = ['0', '1', '2']  # vLLM resource allocation, where each string represents a GPU allocation
PER_VLLM_GPU = len(VLLM_CONFIG[0].split(','))  # Number of GPUs allocated per vLLM task

# Ensure all vLLM configurations have the same number of assigned GPUs
assert len(set([len(v.split(',')) for v in VLLM_CONFIG])) == 1, "every vllm same"

# Training hyperparameters
FORMAT_WEIGHT = 1.0  # Weight for format matching
ACCURACY_WEIGHT = 1.0  # Weight for accuracy

BETA = 0.0  # KL divergence loss weight for reference model (used for loss control)
EPSILON = 0.2  # PPO algorithm clip ratio, controlling the update magnitude of the policy
LR = 3e-6  # Learning rate

# Generation-related parameters
NUM_GENERATIONS = 8  # 
SYS_SET = ("A conversation between User and Assistant. The user asks a question, "
           "and the Assistant solves it. The assistant first thinks about the "
           "reasoning process in the mind and then provides the user with the answer. "
           "The reasoning process and answer are enclosed within <think> </think> "
           "and <answer> </answer> tags, respectively, i.e., <think> reasoning process"
           " here </think><answer> answer here </answer>")

SYS_SETS = [SYS_SET for _ in range(NUM_GENERATIONS)]  # Duplicate system instruction template to ensure consistency across generation tasks

# Ensure MAX_NUM_SEQ is divisible by NUM_GENERATIONS
assert MAX_NUM_SEQ % NUM_GENERATIONS == 0

# Training and testing datasets
TRAIN_DATASET = "xiaodongguaAIGC/X-R1-7500"  # Training dataset
TEST_DATASET = "HuggingFaceH4/aime_2024"  # Testing dataset

# Predefined assistant role token
ASSISTANT_TOKEN = 'assistant' # 
