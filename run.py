import gradio as gr
import time
from get_completion import get_completion, get_stream_completion

# 用于gr.ChatInterface()的输入
def get_response(message, history):
    response = get_completion(prompt=message, api_key= "sk-zk23595cc31f912feed99fd601663df6087681b41de060c3", base_url="https://api.zhizengzeng.com/v1")
    return response

def get_stream_response(message, history):
    response = ""
    for chunk in get_stream_completion(prompt=message, api_key= "sk-zk23595cc31f912feed99fd601663df6087681b41de060c3", base_url="https://api.zhizengzeng.com/v1"):
        time.sleep(0.1)
        response += chunk
        yield response

gr.ChatInterface(fn=get_stream_response, type="messages").launch()