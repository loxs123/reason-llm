import requests
import json

# 设置API的URL和headers
url = "https://api.siliconflow.cn/v1/chat/completions"  # 请确认这个URL是否正确
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-zdiswkwbvrxgmzvfmmurmdtmbcvnzxxpyhrqkkqpdbapryhv"  # 请替换成你的API密钥
}


# 发送请求并处理流式响应
def stream_response():
    while True:

        text = input("请输入：")

        # 构造请求的payload
        data = {
            "model": "deepseek-ai/DeepSeek-R1",  # 模型名称
            "messages": [
                {"role": "system", "content": "You are a the most powerful math expert. Please solve the problems with deep resoning. You are careful and always recheck your conduction. You will never give answer directly until you have enough confidence. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000."},
                {"role": "user", "content": text}
            ],
            "stream": True,  # 开启流式输出
            "max_tokens": 8192,
            "temperature": 0.6

        }

        # 使用 requests.post 发送请求，并设置 stream=True
        response = requests.post(url, headers=headers, json=data, stream=True)

        # 检查响应状态码
        if response.status_code != 200:
            print(f"请求失败，状态码: {response.status_code}")
            print(response.text)
            return

        # 逐行读取流式响应
        for line in response.iter_lines():
            if line:  # 过滤掉空行
                # 去掉每行开头的 "data: " 前缀
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    line_str = line_str[6:]  # 去掉 "data: " 前缀

                if line_str == '[DONE]':
                    print(line_str)
                else:
                    chunk = json.loads(line_str)
                    # 提取并打印内容
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        content = chunk["choices"][0].get("delta", {}).get("content", "")
                        print(content, end="", flush=True)  # 流式打印

        print()

# 调用函数
if __name__ == "__main__":
    stream_response()