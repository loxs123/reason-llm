import ray
import os
from transformers import AutoTokenizer
from reason_llm.config import *

if GENERATE_BACKEND == 'transformers':
    from transformers import AutoModelForCausalLM
    @ray.remote(num_gpus=GENERATE_PER_WORKER_GPU)
    class GenerateWorker:
        def __init__(self, model_dir, gpu_ids):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        def generate(self, prompts, sampling_params):
            temperature = sampling_params.get("temperature", 0.1)
            if temperature > 0:
                _sampling_params = dict(
                    do_sample = True,
                    temperature=temperature,
                    max_new_tokens=sampling_params.get("max_new_tokens", 3000),
                )
            else:
                _sampling_params = dict(
                    do_sample = False,
                    max_new_tokens=sampling_params.get("max_new_tokens", 3000),
                )
            
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
                for p in prompts
            ]
            model_inputs = self.tokenizer(formatted_prompts,padding=True, truncation=True, return_tensors="pt").to(self.llm.device)
            generated_ids = self.llm.generate(**model_inputs, **_sampling_params)[:, model_inputs.input_ids.shape[1]: ]
            outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return [
                prompt + [{"role": "assistant", "content": output}]
                # prompt + [{"role": "assistant", "content": remove_stutter(output.outputs[0].text)}]
                for prompt, output in zip(prompts, outputs)
            ]

elif GENERATE_BACKEND == 'vllm':
    from vllm import LLM, SamplingParams
    @ray.remote(num_gpus=GENERATE_PER_WORKER_GPU)
    class GenerateWorker:
        def __init__(self, model_dir, gpu_ids):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            self.llm = LLM(
                model_dir,
                max_model_len=MAX_MODEL_LEN,
                trust_remote_code=True,
                tensor_parallel_size=GENERATE_PER_WORKER_GPU,
                enable_prefix_caching=True,
                dtype='bfloat16',
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        def generate(self, prompts, sampling_params):

            _sampling_params = SamplingParams(
                temperature=sampling_params.get("temperature", 0.1),
                max_tokens=sampling_params.get("max_new_tokens", 3000),
            )

            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
                for p in prompts
            ]
            outputs = self.llm.generate(formatted_prompts, 
                                        sampling_params=_sampling_params,
                                        use_tqdm=False)
            return [
                prompt + [{"role": "assistant", "content": output.outputs[0].text}]
                # prompt + [{"role": "assistant", "content": remove_stutter(output.outputs[0].text)}]
                for prompt, output in zip(prompts, outputs)
            ]

elif GENERATE_BACKEND == 'lmdeploy':
    from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
    @ray.remote(num_gpus=GENERATE_PER_WORKER_GPU)
    class GenerateWorker:
        def __init__(self, model_dir, gpu_ids):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            backend_config = TurbomindEngineConfig(
                dtype = 'bfloat16',
                tp=GENERATE_PER_WORKER_GPU,
                enable_prefix_caching = True,
            )
            self.llm = pipeline(model_dir, backend_config=backend_config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        def generate(self, prompts, sampling_params):
            temperature = sampling_params.get("temperature", 0.1)
            do_sample = temperature > 0
            _sampling_params = GenerationConfig(
                do_sample = do_sample,
                temperature=temperature,
                max_new_tokens=sampling_params.get("max_new_tokens", 3000),
            )

            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
                for p in prompts
            ]
            outputs = self.llm(formatted_prompts, gen_config=_sampling_params)
            return [
                prompt + [{"role": "assistant", "content": output.text}]
                # prompt + [{"role": "assistant", "content": remove_stutter(output.outputs[0].text)}]
                for prompt, output in zip(prompts, outputs)
            ]
elif GENERATE_BACKEND == 'sglang':
    import sglang as sgl
    @ray.remote(num_gpus=GENERATE_PER_WORKER_GPU)
    class GenerateWorker:
        def __init__(self, model_dir, gpu_ids):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

            self.llm = sgl.Engine(model_path=model_dir,tp_size = GENERATE_PER_WORKER_GPU)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        def generate(self, prompts, sampling_params):
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
                for p in prompts
            ]
            outputs = self.llm.generate(formatted_prompts, sampling_params)
            return [
                prompt + [{"role": "assistant", "content": output['text']}]
                # prompt + [{"role": "assistant", "content": remove_stutter(output.outputs[0].text)}]
                for prompt, output in zip(prompts, outputs)
            ]
else:
    raise ValueError(f"Unsupported GENERATE_BACKEND: {GENERATE_BACKEND}")
