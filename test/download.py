# import os

# os.environ['HF_HOME'] = '/root/shared-storage'
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# from transformers import AutoModelForCausalLM, AutoTokenizer

# AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
# AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')


#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')