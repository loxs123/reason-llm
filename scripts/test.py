import torch

import torch


# 示例用法
data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float32)
mask = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=torch.float32)

z_scores = masked_z_score(data, mask, fill_value=0)
print(z_scores)