import time
from grpo_vllm import TrainingSamplingCoordinator

if __name__ == "__main__":
    coordinator = TrainingSamplingCoordinator()
    # 持续运行训练-采样循环
    for i in range(10):
        print("\n=== 开始新的迭代周期 ===")
        start_time = time.time()
        coordinator.run_cycle()
        cycle_time = time.time() - start_time
        print(f"周期完成，耗时: {cycle_time//3600:.0f}h {(cycle_time%3600)//60:.0f}m {cycle_time%60:.2f}s")



