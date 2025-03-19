import re
import matplotlib.pyplot as plt
import pandas as pd
from reason_llm.config import *

step = INT_NUM * REP_NUM // (GPU_NUM * per_device_train_batch_size * gradient_accumulation_steps)

with open('nohup.out') as f:
    data = f.read()
    y1 = re.findall(r'平均正确率：(.*?)，多数投票正确率：(.*?)，平均奖励：(.*?)，平均长度：(.*?)\n已写入.*?条样本到缓冲区', data)
    y20 = re.findall(r'平均正确率：(.*?)，多数投票正确率：(.*?)，平均奖励：(.*?)，平均长度：(.*?)\n\n=== 开始新的迭代周期 ===', data)
    y2 = y20 + re.findall(r'平均正确率：(.*?)，多数投票正确率：(.*?)，平均奖励：(.*?)，平均长度：(.*?)\n周期完成', data)

# 定义不同的数据计算方式
data_options = [
    ([float(_y[2]) - float(_y[0]) for _y in y1], 'Train Format Reward'),
    ([float(_y[0]) for _y in y1], 'Train Accuracy Reward'),
    ([float(_y[1]) for _y in y1], 'Train Major Vote Accuracy'),
    ([float(_y[3]) for _y in y1], 'Train Length'),
    ([float(_y[2]) - float(_y[0]) for _y in y2], 'Test Format Reward'),
    ([float(_y[0]) for _y in y2], 'Test Accuracy Reward'),
    ([float(_y[1]) for _y in y2], 'Test Major Accuracy'),
    ([float(_y[3]) for _y in y2], 'Test Length')
]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 创建2行4列的子图
fig.suptitle('Training and Testing Metrics Analysis')  # 设置整体标题

for ax, (y, title) in zip(axes.flat, data_options):
    y_series = pd.Series(y)
    y_smooth = y_series.rolling(window=5).mean()
    x = list(range(len(y)))
    x = [_x * step for _x in x]
    
    ax.plot(x, y, label='Original Data', marker='.')
    ax.plot(x, y_smooth, label='Smoothed Data', marker='.')
    ax.set_title(title)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Value')
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整子图间距
plt.savefig('metrics_analysis.png', dpi=300)
plt.show()
