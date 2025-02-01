import ray

@ray.remote
class Counter:
    def __init__(self):
        self.count = 0  # 共享变量

    def increment(self):
        self.count += 1
        return self.count

# 创建一个 Actor 作为共享对象
counter = Counter.remote()

# 多个任务调用
print(ray.get(counter.increment.remote()))  # 1
print(ray.get(counter.increment.remote()))  # 2
