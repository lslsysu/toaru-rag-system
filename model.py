from typing import List, Mapping, Any, Optional, Iterator
from openai import OpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain.callbacks.manager import CallbackManagerForLLMRun

class CustomChatModel(BaseChatModel):
    api_key: str
    base_url: str
    model: str
    temperature: float

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        prompt = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                # 其他消息类型
                role = "others"
            prompt += f"{role}: {message.content}\n"
        return prompt.strip()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None
    ) -> ChatResult:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        prompt = self._convert_messages_to_prompt(messages)
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=False
        )
        output = response.choices[0].message.content
        message = AIMessage(content=output)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None
    ) -> Iterator[ChatGenerationChunk]:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        prompt = self._convert_messages_to_prompt(messages)

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=True
        )
        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield ChatGenerationChunk(message=AIMessageChunk(content=content))

    @property
    def _llm_type(self) -> str:
        return "custom-chat"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "temperature": self.temperature,
            "model": self.model,
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # API configuration
    API_KEY = os.getenv("API_KEY")
    BASE_URL = 'https://api.zhizengzeng.com/v1'


    chat = CustomChatModel(api_key= API_KEY, base_url=BASE_URL, model = 'gpt-3.5-turbo', temperature=0.0)
    response = chat.invoke([HumanMessage(content="hello!")])
    print(response.content)

    for chunk in chat.stream("hello!"):
        print(chunk.content, end="|")