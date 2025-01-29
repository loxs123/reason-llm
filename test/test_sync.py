# import os
# import time

# t1 = os.path.getmtime(r'C:\Users\asndj\Desktop\grpo-vllm\grpo_vllm\config.json') # ms
# t2 = os.path.getmtime(r'C:\Users\asndj\Desktop\grpo-vllm\grpo_vllm\model.safetensors') # ms

# print(t1)
# print(t2)
# # print(time.time())

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)