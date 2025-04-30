import os
import json
import gc
import numpy as np
import torch
import random
import time
import re
import ray
import glob

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

from reason_llm.utils import apply_lora, get_per_token_logps, remove_stutter
from reason_llm.generate import GenerateWorker
from reason_llm.config import *
from reason_llm.reward_fn import *

def load_data(data_path):
    if os.path.exists(data_path):
        return load_from_disk(data_path)
    return load_dataset(data_path)

def average_length(messages, tokenizer):
    total_len = 0
    for msg in messages:
        total_len += len(tokenizer.tokenize(msg[-1]['content']))
    return total_len / len(messages)

def average(values):
    return sum(values) / len(values)

class LLMTrainingManager:
    def __init__(self):
        self.train_index = START_TRAIN_IDX
        self.avg_accuracy = []
        self.rewards = []
        self.majority_accuracy = []
        self.response_lengths = []
        self.print_config_file()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.train_dataset = load_data(TRAIN_DATASET)['train'].shuffle(seed=42)
        self.test_datasets = [load_data(ds_name) for ds_name in TEST_DATASETS]
        print('Train Set size: ', len(self.train_dataset))
        self.start_generation_workers()

    def print_config_file(self):
        with open(config_file) as f:
            print('#' * 10, 'CONFIG_START', '#' * 10)
            print(f.read())
            print('#' * 10, 'CONFIG_END', '#' * 10)

    def _get_next_buffer_id(self):
        buffer_files = sorted(glob.glob(os.path.join(data_dir, 'buffer*.json')), reverse=True)
        if not buffer_files:
            return 0
        return int(re.findall('buffer(.*?).json', buffer_files[0])[0]) + 1

    def start_generation_workers(self):
        apply_lora(model_dir)
        model_path = os.path.join(model_dir, "merge")
        if not os.path.exists(model_path):
            model_path = model_dir

        self.generation_workers = [
            GenerateWorker.remote(model_path, gpu_ids)
            for gpu_ids in GENERATE_GPU_CONFIG
        ]

    def _parallel_generate(self, prompts, sampling_params):
        if not self.generation_workers:
            raise RuntimeError("No available generation workers!")

        chunk_size = (len(prompts) + len(self.generation_workers) - 1) // len(self.generation_workers)
        prompt_chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]

        futures = [worker.generate.remote(chunk, sampling_params)
                   for worker, chunk in zip(self.generation_workers, prompt_chunks)]
        results = ray.get(futures)

        return [msg for result in results for msg in result]

    def _compute_combined_rewards(self, completions, references):
        acc_rewards, acc_rates, major_acc_rates = accuracy_reward(completions, references)
        format_rewards = format_reward(completions)

        total_rewards = [a * ACCURACY_WEIGHT + f * FORMAT_WEIGHT
                         for a, f in zip(acc_rewards, format_rewards)]
        return total_rewards, acc_rates, major_acc_rates

    def print_evaluation_metrics(self, dataset_name):
        print(f'[{dataset_name}] Average Acc: {average(self.avg_accuracy)}，'
              f'Major Acc: {average(self.majority_accuracy)}，'
              f'Average Reward: {average(self.rewards)}，'
              f'Average Length: {average(self.response_lengths)}')

    def reset_evaluation_metrics(self):
        self.avg_accuracy.clear()
        self.majority_accuracy.clear()
        self.rewards.clear()
        self.response_lengths.clear()

    def _generate_buffer_samples(self, prompt_batches, reference_solutions, mode='train'):
        sampling_params = {
            'temperature': 1.0 if mode == 'train' else 0.0,
            'max_new_tokens': TRAIN_MAX_GENERATE_LEN if mode == 'train' else EVAL_MAX_GENERATE_LEN
        }

        num_generations = NUM_GENERATIONS if mode == 'train' else 1
        prompts = [msg for batch in prompt_batches for msg in batch]
        completions = self._parallel_generate(prompts, sampling_params)
        rewards, acc_rates, majority_acc_rates = self._compute_combined_rewards(completions, reference_solutions)

        self.rewards += rewards
        self.avg_accuracy += acc_rates
        self.majority_accuracy += majority_acc_rates

        buffer_samples = []
        for i in range(0, len(completions), num_generations):
            self.response_lengths.append(average_length(completions[i:i + num_generations], self.tokenizer))
            sliced_rewards = np.array(rewards[i:i + num_generations])
            acc_rate = acc_rates[i // num_generations]

            if USE_DYNAMIC_BATCH and (acc_rate <= 0.01 or acc_rate >= 0.99):
                continue

            advantages = sliced_rewards - sliced_rewards.mean()
            if SCALE_ADV_WITH_STD:
                advantages /= sliced_rewards.std() + 1e-8

            for msg, adv, r in zip(completions[i:i + num_generations], advantages, sliced_rewards):
                buffer_samples.append({
                    "completion": msg,
                    "advantage": adv.item(),
                    "reward": r.item(),
                    "label": reference_solutions[i // num_generations],
                })

        return buffer_samples

    def sample_training_data(self):
        print("\n[Sample Stage]")
        sample_time = time.time()
        self.buffer_samples = []
        batched_prompt_msgs = []
        batched_solutions = []
        samples_per_batch = MAX_NUM_SEQ // NUM_GENERATIONS

        while len(self.buffer_samples) < INT_NUM:
            raw_row = self.train_dataset[self.train_index % len(self.train_dataset)]
            normalized_sample = {k.lower(): v for k, v in raw_row.items()}
            batched_prompt_msgs.append(build_msgs(normalized_sample, dataset=TRAIN_DATASET, num_generations=NUM_GENERATIONS))
            batched_solutions.append(build_sol(normalized_sample, dataset=TRAIN_DATASET))

            if (self.train_index + 1) % samples_per_batch == 0:
                self.buffer_samples += self._generate_buffer_samples(batched_prompt_msgs, batched_solutions)
                batched_prompt_msgs.clear()
                batched_solutions.clear()
            self.train_index += 1

        self.print_evaluation_metrics(TRAIN_DATASET)
        self.buffer_samples = self.buffer_samples[:INT_NUM]
        print('End Train Index:', self.train_index, 'Sample Time:', time.time() - sample_time)

    def save_buffer_to_file(self):
        random.shuffle(self.buffer_samples)
        self.reset_evaluation_metrics()
        buffer_id = self._get_next_buffer_id()
        buffer_path = os.path.join(data_dir, f'buffer{buffer_id:05d}.json')
        with open(buffer_path, "w") as f:
            json.dump(self.buffer_samples, f, ensure_ascii=False, indent=2)
        self.buffer_samples.clear()

    def terminate_generation_workers(self):
        for worker in self.generation_workers:
            ray.kill(worker)
        self.generation_workers = []
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    def run_training_script(self):
        print("\n[Train Stage]")
        config_path = "deepspeed_zero2.yaml" if GPU_NUM == 1 else "deepspeed_zero3.yaml"
        os.system(f"CUDA_VISIBLE_DEVICES={GPU} accelerate launch "
                  f"--config_file {current_dir}/reason_llm/ds_cfgs/{config_path} "
                  f"{current_dir}/reason_llm/trainer.py")

    def evaluate_on_test_datasets(self):
        print("\n[Test Stage]")
        for dataset, dataset_name in zip(self.test_datasets, TEST_DATASETS):
            batched_prompt_msgs = []
            batched_solutions = []
            self.reset_evaluation_metrics()
            samples_per_batch = MAX_NUM_SEQ

            if 'train' in dataset:
                dataset = dataset['train']
            elif 'test' in dataset:
                dataset = dataset['test']

            for i in range(len(dataset)):
                normalized_sample = {k.lower(): v for k, v in dataset[i].items()}
                batched_prompt_msgs.append(build_msgs(normalized_sample, dataset=dataset_name, num_generations=1))
                batched_solutions.append(build_sol(normalized_sample, dataset=dataset_name))

                if (i + 1) % samples_per_batch == 0:
                    self._generate_buffer_samples(batched_prompt_msgs, batched_solutions, 'test')
                    batched_prompt_msgs.clear()
                    batched_solutions.clear()

            if batched_prompt_msgs:
                self._generate_buffer_samples(batched_prompt_msgs, batched_solutions, 'test')

            self.print_evaluation_metrics(dataset_name)
            self.reset_evaluation_metrics()

    def run_training_cycle(self):
        self.sample_training_data()
        self.terminate_generation_workers()
        self.save_buffer_to_file()
        self.run_training_script()
        self.start_generation_workers()
        self.evaluate_on_test_datasets()
