import os
import re

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_dir = os.path.join(current_dir, "model")
log_dir = os.path.join(current_dir, "log")
buffer_file = os.path.join(current_dir, "data", "buffer.json")

per_device_train_batch_size = 4 # 大一点的batch_size
gradient_accumulation_steps = 64 

MAX_MODEL_LEN = 8096
MAX_NUM_SEQ = 32
INT_NUM = 1024
REP_NUM = 1

GPU = "0"
GPU_NUM = len(GPU.split(","))

FORMAT_WEIGHT = 0.0 # only acc reward
ACCURACY_WEIGHT = 1.0 

BETA = 0.0 # ref model kl loss
EPSILON = 0.2 # ppo clip radio
LR = 3e-6

NUM_GENERATIONS = 8
SYS_SET = ("You are a seasoned science expert, recognized for your thorough analysis and precise calculations."
          "Work through the given problem step by step, ensuring every detail is carefully considered. "
          "Present your solution inside <think></think> tags, with the final result enclosed in \\boxed{}.")

SYS_SETS = [SYS_SET for _ in range(NUM_GENERATIONS)]

assert MAX_NUM_SEQ % NUM_GENERATIONS == 0

TEST_DATASET = 'BBexist/AIME25'
TRAIN_DATASET = 'di-zhang-fdu/AIME_1983_2024'

# https://zhuanlan.zhihu.com/p/21465667399
# 需要调整一下 tokenizer_config.json 中的字段，参照博客内容
ASSISTANT_TOKEN = '<｜Assistant｜>' # for deepseek qwen 
