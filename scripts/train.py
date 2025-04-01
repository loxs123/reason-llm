import time
from reason_llm import TrainingSamplingCoordinator
from reason_llm import config as cfg

if __name__ == "__main__":
    coordinator = TrainingSamplingCoordinator()
    coordinator.test_model()
    # 持续运行训练-采样循环
    for i in range(cfg.ITER_NUM):
        print(f'[{i}] Iter start')
        start_time = time.time()
        coordinator.generate_samples()
        coordinator.del_vllm_workers()
        # coordinator.compute_logp()
        coordinator.save_samples()
        coordinator.train_model() # lora：base model_dir lora lora
        coordinator.init_vllm_workers() # merge : base + lora
        if (i + 1) % cfg.TEST_FREQ == 0: coordinator.test_model()
        cycle_time = time.time() - start_time
        print(f"Iter Time : {cycle_time//3600:.0f}h {(cycle_time%3600)//60:.0f}m {cycle_time%60:.2f}s")

