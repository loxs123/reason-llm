
import re
from collections import Counter
# TODO 加入一些更精细化的奖励

#extract the integer answer modulo 1000 from the boxes
def extract_boxed_text(text):
    matches = re.findall(r'oxed{(.*?)}', text)
    if not matches:
        return -1
    content = matches[-1]
    if content.isdigit():
        num = int(content)
    else:
        nums = re.findall(r'\d+', content)
        if not nums:
            return -1
        num = int(nums[-1])
    return num % 1000

def length_reward_fn(msgs):

    # clip 0 - 1
    
    base = (len(msgs[-1]['content']) - 5000) / 25000

    if base < 0: base = 0.0
    if base > 1: base = 1.0

    return 1.0 - base


def correct_reward_fn(msgs, labelset):
    p = extract_boxed_text(msgs[-1]['content'])
    correct = p in labelset
    if correct: return 1.0, p
    else: return 0.0, p

def _reward_fn(msgs, label):
    r1, p = correct_reward_fn(msgs, label)
    r2 = length_reward_fn(msgs)
    if r1 > 0.5: return r1, p
    else: return r1, p

def group_reward_fn(prompts=None, completions=None, label=None):
    labelset = set()
    if type(label) is str:
        nums = re.findall(r'\d+', label)
        for num in nums: labelset.add(int(num))
    if type(label) is int:
        labelset.add(label)

    rewards = []

    predicts = []
    for c in completions:
        r, p = _reward_fn(c, labelset)
        rewards.append(r)
        predicts.append(p)

    # 使用 Counter 统计每个元素出现的次数
    counts = Counter(predicts)
    
    # 找到出现次数最多的元素
    me, mc = counts.most_common(1)[0]

    correct = me in labelset
    
    return rewards, correct