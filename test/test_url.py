import requests
import json

import requests

# url = "https://api.siliconflow.cn/v1/chat/completions"

# payload = {
#     "messages": [
#         {
#             "content": "你好",
#             "role": "user"
#         }
#     ],
#     "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
# }
# headers = {
#     "Authorization": "Bearer sk-kcwegjzyevtqxjztlmhjqulbqvmeqgjeymnwdpggxcamkyjh",
#     "Content-Type": "application/json"
# }

# response = requests.request("POST", url, json=payload, headers=headers)

# print(response.text)

# 设置API的URL和headers
url = "https://api.siliconflow.cn/v1/chat/completions"  # 请确认这个URL是否正确
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-kcwegjzyevtqxjztlmhjqulbqvmeqgjeymnwdpggxcamkyjh"  # 请替换成你的API密钥
}


# 发送请求并处理流式响应
def stream_response():
    while True:

        try:
            text = input("请输入：")

            # 构造请求的payload
            data = {
                "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",  # 模型名称
                # "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",  # 模型名称
                "messages": [
                    # {"role": "user", "content": text + '\nIf the final answer is a number larger than 1000, take modulo 1000. please write a python program to solve.'}
                    {"role": "user", "content": text + '\n这是一道数学难题，请输出结果除以1000的余数。'}
                ],
                "stream": True,  # 开启流式输出
                "max_tokens": 16192,
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
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get('content')
                            if content is None:
                                content =delta.get('reasoning_content')
                                
                            print(content, end="", flush=True)  # 流式打印

            print()
        
        except Exception as e:
            break

# 调用函数
if __name__ == "__main__":
    stream_response()