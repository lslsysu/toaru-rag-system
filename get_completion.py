import requests
from langchain_openai import OpenAIEmbeddings
import json

url = "http://localhost:8000/v1/chat/completions"

def get_completion(prompt, provider='zhizengzeng', model='gpt-3.5-turbo', temperature=0.0, top_k=5,
                   api_key=None, base_url=None):
    request = {
                'prompt': prompt,
                'provider': provider,
                'model': model,
                'temperature': temperature,
                'top_k': top_k,
                'api_key': api_key,
                'base_url': base_url,
                'stream' : False
            }
    response = requests.post(url, json=request).json()
    return response["context"]

def get_stream_completion(prompt, provider='zhizengzeng', model='gpt-3.5-turbo', temperature=0.0, top_k=5,
                   api_key=None, base_url=None):
    request = {
                'prompt': prompt,
                'provider': provider,
                'model': model,
                'temperature': temperature,
                'top_k': top_k,
                'api_key': api_key,
                'base_url': base_url,
                'stream' : True
            }
    response = requests.post(url, json=request, stream=True)
    result = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line.strip():
            continue
        # 去掉冗余前缀data: "
        elif line.startswith("data:"):
            line = line[len("data:"):].strip()
        if line == "[DONE]":
            break

        try:
            data = json.loads(line)
            content = data["text"]
            if content:
                yield content
        except json.JSONDecodeError:
            continue

if __name__ == '__main__':

    from dotenv import load_dotenv

    load_dotenv()

    # API configuration
    API_KEY = os.getenv("API_KEY")
    BASE_URL = 'https://api.zhizengzeng.com/v1'

    for chunk in get_stream_completion(prompt='介绍一下欧提努斯', api_key= API_KEY, base_url=BASE_URL):
        print(chunk, end="|", flush=True)