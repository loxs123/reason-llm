
import re

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
    
    base = len(msgs[-1]['content']) - 5000 / 25000

    if base < 0: base = 0.0
    if base > 1: base = 1.0

    return 1.0 - base


def correct_reward_fn(msgs, label):

    labelset = set()
    if type(label) is str:
        nums = re.findall(r'\d+', label)
        for num in nums: labelset.add(int(num))
    if type(label) is int:
        labelset.add(label)
    
    correct = extract_boxed_text(msgs[-1]['content']) in labelset
    if correct: return 1.0
    else: return 0.0


def _reward_fn(msgs, label):
    r1 = correct_reward_fn(msgs, label)
    r2 = length_reward_fn(msgs)
    if r1 > 0.5: return r1 + r2 * 0.2
    else: return r1

def batch_group_reward_fn(prompts=None, completions=None, label=None):
    rewards = []
    for c,l in zip(completions, label):
        for _c in c:
            rewards.append(_reward_fn(_c, l))
    return rewards

def group_reward_fn(prompts=None, completions=None, label=None):
    rewards = []
    for c in completions:
        rewards.append(_reward_fn(c, label))
    return rewards