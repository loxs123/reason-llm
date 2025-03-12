import os
import re

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_dir = os.path.join(current_dir, "model")
log_dir = os.path.join(current_dir, "log")
buffer_file = os.path.join(current_dir, "data", "buffer.json")

per_device_train_batch_size = 4 # 大一点的batch_size
gradient_accumulation_steps = 64 

system_setting_file = os.path.join(current_dir, "reason_llm/system_setting.txt")
with open(system_setting_file) as f:
    d = f.read()
    system_settings = re.findall(r'```text\n(.*?)\n```', d, re.S)

MAX_MODEL_LEN = 8096
MAX_NUM_SEQ = 32
INT_NUM = 1024
REP_NUM = 1

GPU = "0"
GPU_NUM = len(GPU.split(","))

FORMAT_WEIGHT = 0.0 # only 
ACCURACY_WEIGHT = 1.0 

BETA = 0.0 # ref model kl loss
EPSILON = 0.2 # ppo clip radio
LR = 3e-6

NUM_GENERATIONS = len(system_settings)
assert MAX_NUM_SEQ % NUM_GENERATIONS == 0

TEST_DATASET = 'BBexist/AIME25'
TRAIN_DATASET = 'di-zhang-fdu/AIME_1983_2024'

# https://zhuanlan.zhihu.com/p/21465667399
# 需要调整一下tokenizer_config.json 中的字段，参照博客内容
ASSISTANT_TOKEN = '<｜Assistant｜>' # for deepseek qwen 

## for qwen-7b-config
# TRAIN_DATASET = "xiaodongguaAIGC/X-R1-7500"
# TEST_DATASET = "HuggingFaceH4/aime_2024"
# FORMAT_WEIGHT = 1.0
# MAX_MODEL_LEN = 2048
# ASSISTANT_TOKEN = 'assistant' # for qwen
