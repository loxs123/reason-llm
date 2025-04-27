# # for test 
# import os
# from vllm import LLM, SamplingParams
# from datasets import load_dataset,load_from_disk
# import time
# import copy

# def load_data_from_disk_or_hf(data_name):
#     if os.path.exists(data_name):
#         return load_from_disk(data_name)
#     return load_dataset(data_name)

# class VLLMWorker:
#     def __init__(self, model_dir, gpu_ids):
#         os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
#         self.llm = LLM(
#             model_dir,
#             max_model_len=2048,
#             trust_remote_code=True,
#             tensor_parallel_size=1,
#             enable_prefix_caching=True,
#             dtype='bfloat16',
#         )
#         self.tokenizer = copy.deepcopy(self.llm.get_tokenizer())

#     def generate(self, prompts, sampling_params):
#         formatted_prompts = [
#             self.tokenizer.apply_chat_template(
#                 p, tokenize=False, add_generation_prompt=True
#             )
#             for p in prompts
#         ]
#         outputs = self.llm.generate(formatted_prompts, sampling_params=sampling_params)
#         return [
#             prompt + [{"role": "assistant", "content": output.outputs[0].text}]
#             for prompt, output in zip(prompts, outputs)
#         ]

# if __name__ == '__main__':
#     ds = load_data_from_disk_or_hf('HuggingFaceH4/MATH-500')['test']
#     llm = VLLMWorker('/root/workspace/model', '0')

#     sampling_params = SamplingParams(
#         temperature=1.0,
#         max_tokens=2048,
#     )

#     start_time = time.time()

#     parallel_seq_num = 1024
#     msgs = []
    
#     for i, item in enumerate(ds):
#         msgs += [[{'role': 'user', 'content': item['problem']}] for _ in range(8)]
#         if len(msgs) == parallel_seq_num:
#             llm.generate(msgs, sampling_params)
#             msgs.clear()
#         if i == 255: break

#     print('use_time: ', time.time() - start_time)


from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("zwhe99/DeepMath-103K")


