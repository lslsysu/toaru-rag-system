from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from rag import RAGChain, indexing
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Generator, Optional
from typing import AsyncGenerator
import json
import torch
import uvicorn
import os
import sys
# 导入功能模块目录
sys.path.append("./")

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
app = FastAPI(lifespan=lifespan)

# 允许跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# request
class Item(BaseModel):
    prompt : str
    provider : str = "zhizengzeng"
    model : str = "gpt-3.5-turbo"
    temperature : float = 0.0
    api_key : str = None
    base_url : str = None
    top_k : int = 5 # Top K
    stream : bool = False

# 响应体
class ChatCompletionResponse(BaseModel):
    model: str
    context: str

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def get_response(item: Item):
    vectordb = indexing()
    rag_chain = RAGChain(model=item.model, vectordb=vectordb, top_k=item.top_k,
                                 provider=item.provider, api_key=item.api_key, base_url=item.base_url)
    if not item.stream:
        answer = rag_chain.answer(question=item.prompt)
        return ChatCompletionResponse(model=item.model, context=answer)
    else:
        async def event_stream() -> AsyncGenerator[str, None]:
            for chunk in rag_chain.answer_stream(question=item.prompt):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)