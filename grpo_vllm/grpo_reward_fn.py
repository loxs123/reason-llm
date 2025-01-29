
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

def reward_fn(prompts=None, completions=None, label=None):
    rewards = []
    for c,l in zip(completions, label):
        for _c in c:
            if extract_boxed_text(_c[-1]['content']) == int(l):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
    return rewards



