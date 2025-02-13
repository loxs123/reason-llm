
import re
from collections import Counter


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
    return num

def format_reward_fn(msgs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = 
    # pattern = r"<think>.*?</think>.*?oxed{(.*?)}"
    think_num = len(re.findall(r"<think>.*?</think>", msgs[-1]['content'], re.DOTALL))
    find_num = len(re.findall(r"<think>.*?</think>.*?oxed{.*?}", msgs[-1]['content'], re.DOTALL))

    if find_num == 1 and think_num == 1:
        return 1.0
    elif think_num > 1:
        return 0.5
    else:
        return 0.0

def length_reward_fn(msgs):

    # clip 0 - 1
    
    content = msgs[-1]['content']

    words = set()
    for i in range(0, len(content) - 20):
        words.add(content[i:i+20])
    
    return min((len(words) + 20) / 10000, 1.0) # 越长越好

    # base = (len(msgs[-1]['content']) - 5000) / 25000

    # if base < 0: base = 0.0
    # if base > 1: base = 1.0

    # return 1.0 - base

# def llm_score(msgs):
#     
    
#     return 1.0


def correct_reward_fn(msgs, labelset):
    p = extract_boxed_text(msgs[-1]['content'])
    correct = p in labelset
    if correct: return 1.0, p
    else: return 0.0, p

def llm_reward_fn(score_msgs):
    score = re.findall(r'oxed{(.*?)}', score_msgs[-1]['content'])
    if len(score) == 0:
        return 0.5
    else:
        try:
            s = float(score[-1])
            s = max(min(s, 10), 0) / 10
        except:
            s = 0.5
        
        return s


def _reward_fn(msgs, score_msgs, label):
    r1, p = correct_reward_fn(msgs, label) # 回答正确
    # r2 = format_reward_fn(msgs) # 格式
    # r3 = length_reward_fn(msgs) # 格式
    r4 = llm_reward_fn(score_msgs)

    return (r1 + r4) / 2, p

def group_reward_fn(prompts=None, completions=None, label=None, scores = None):
    labelset = set()
    if type(label) is str:
        nums = re.findall(r'\d+', label)
        for num in nums: labelset.add(int(num))
    if type(label) is int:
        labelset.add(label)

    rewards = []

    predicts = []
    acc_cnt = 0
    for c,s in zip(completions, scores):
        r, p = _reward_fn(c, s, labelset)
        rewards.append(r)
        if p != -1:
            predicts.append(p)
        if p in labelset:
            acc_cnt += 1
            
    if len(predicts)== 0:
        predicts.append(-1)

    # 使用 Counter 统计每个元素出现的次数
    counts = Counter(predicts)
    
    # 找到出现次数最多的元素
    me, mc = counts.most_common(1)[0]

    correct = me in labelset
    
    return rewards, correct, acc_cnt / len(rewards)
