import time
from reason_llm import LLMTrainingManager
from reason_llm import config as cfg

if __name__ == "__main__":
    coordinator = LLMTrainingManager()
    coordinator.evaluate_on_test_datasets()
    for i in range(cfg.ITER_NUM):
        print(f'[{i}] Iter start')
        start_time = time.time()
        coordinator.sample_training_data()
        coordinator.terminate_generation_workers()
        coordinator.save_buffer_to_file()
        coordinator.run_training_script()
        coordinator.start_generation_workers()
        if (i + 1) % cfg.TEST_FREQ == 0:
            coordinator.evaluate_on_test_datasets()
        cycle_time = time.time() - start_time
        print(f"Iter Time : {cycle_time//3600:.0f}h {(cycle_time%3600)//60:.0f}m {cycle_time%60:.2f}s")
