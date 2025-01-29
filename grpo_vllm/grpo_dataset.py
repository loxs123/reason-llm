import torch.utils.data as data
import json
import time
import os

class GRPODataset(data.Dataset):
    def __init__(self, filename, max_retries=10, retry_interval=0.5):
        super(GRPODataset, self).__init__()

        self.filename = filename
        self.max_retries = max_retries
        self.retry_interval = retry_interval
    
    def last_change_mtime(self):
        mod_time = os.path.getmtime(self.filename)
        return mod_time
    
    def _wait_for_file(self):
        """等待文件可读"""
        retries = 0
        while retries < self.max_retries:
            try:
                with open(self.filename, "r") as f:
                    return json.load(f)  # 试着读取 JSON，如果成功，则文件稳定
            except (json.JSONDecodeError, OSError):
                pass  # 文件可能正在被写入，重试
            retries += 1
            time.sleep(self.retry_interval)
        raise RuntimeError(f"无法读取 {self.filename}，可能被持续写入")

    def __len__(self):
        data = self._wait_for_file()
        return len(data)

    def __getitem__(self, index):
        data = self._wait_for_file()
        return data[index % len(data)]
