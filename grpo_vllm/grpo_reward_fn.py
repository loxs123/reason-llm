
import re
from collections import Counter
# from Levenshtein import ratio as levenshtein_ratio
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

def format_reward_fn(msgs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>.*?oxed{(.*?)}.*?$"
    match = re.match(pattern, msgs[-1]['content'], re.DOTALL) 
    return 1.0 if match else 0.0

def length_reward_fn(msgs):

    # clip 0 - 1
    
    base = (len(msgs[-1]['content']) - 5000) / 25000

    if base < 0: base = 0.0
    if base > 1: base = 1.0

    return 1.0 - base

def llm_score(msgs):
    prompt = """**Scoring Criteria:**

Please score the given text on a scale from 0 to 1, where the score represents the degree to which the text lacks clarity, logical flow, organization, and contains redundancy. A lower score indicates the presence of more of these issues.

1. **Logical Clarity**: Evaluate whether the text presents its ideas in a coherent and understandable way.
   - *Poor logic*: The text is difficult to follow or lacks clear reasoning, making it hard to understand.

2. **Organization**: Assess whether the text is well-structured, with information presented in a logical order.
   - *Disorganized*: The text lacks clear structure, with ideas scattered and presented in a disjointed or confusing manner.

3. **Conciseness**: Check for unnecessary repetition or redundancy in the text.
   - *Repetition*: The same or similar points are restated multiple times, leading to redundancy and unnecessary length.

Please provide a score based on the severity of these issues, with lower scores indicating more significant problems in terms of logic, organization, and repetition.

The score should be displayed in the following format:  
\[
\\boxed{score}
\]"""
    
    
    return 1.0
    

def correct_reward_fn(msgs, labelset):
    p = extract_boxed_text(msgs[-1]['content'])
    correct = p in labelset
    if correct: return 1.0, p
    else: return 0.0, p

def _reward_fn(msgs, label):
    r1, p = correct_reward_fn(msgs, label)
    r2 = format_reward_fn(msgs)
    
    return (r1 + r2) / 2, p

def group_reward_fn(prompts=None, completions=None, label=None):
    labelset = set()
    if type(label) is str:
        nums = re.findall(r'\d+', label)
        for num in nums: labelset.add(int(num))
    if type(label) is int:
        labelset.add(label)

    rewards = []

    predicts = []
    acc_cnt = 0
    for c in completions:
        r, p = _reward_fn(c, labelset)
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
