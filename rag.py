from langchain.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from model import CustomChatModel
import os
from dotenv import load_dotenv

load_dotenv()

# API configuration
API_KEY = os.getenv("API_KEY")
BASE_URL = 'https://api.zhizengzeng.com/v1'

# for LangChain
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL

# indexing
def indexing(persist_path='vectordb/chroma'):

    embeddings = OpenAIEmbeddings(api_key=API_KEY, base_url=BASE_URL)

    # if persist_path not exist，create vectordb firstly
    if not os.path.exists(persist_path):
        # LOAD and unique
        # toaru
        loader = DirectoryLoader(path="books/toaru/", glob="**/OEBPS/Text/*.xhtml", loader_cls=UnstructuredHTMLLoader)
        documents = loader.load()
        unique_documents = []
        seen_contents = set()

        for doc in documents:
            content = doc.page_content.strip()
            if content not in seen_contents:
                unique_documents.append(doc)
                seen_contents.add(content)

        print(f"原始文档数量: {len(documents)}")
        print(f"去重后文档数量: {len(unique_documents)}")

        # SPLIT
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(unique_documents)

        # EMBED and vectordb
        os.makedirs(persist_path, exist_ok=True)

        vectordb = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_path)
        vectordb.persist()
        return vectordb
    # load persist_path from vectordb
    else:
        print("persist_path已存在")
        vectordb = Chroma(persist_directory=persist_path, embedding_function=embeddings)
        return vectordb

# Retrieval and Generation
class RAGChain:
    """rag chain"""
    def __init__(self, model, temperature=0.0, top_k=4, vectordb=None, provider=None, api_key=None, base_url=None, template=None):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.vectordb = vectordb
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.template = """依据下面的轻小说中截取的段落回答问题
        小说内容：{context}

        问题：{question}
        """

    def answer(self, question=None):
        """construct rag chain and answer question"""

        # llm
        chat = CustomChatModel(api_key=self.api_key, base_url=self.base_url, model=self.model, temperature = self.temperature)
        # retriever
        retriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={'k': self.top_k})  # 默认similarity，k=4

        # print(retriever.invoke(question))
        # prompt template
        prompt = ChatPromptTemplate.from_template(self.template)

        # chain
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | chat
        )

        answer = rag_chain.invoke(question)
        return answer.content

    def answer_stream(self, question=None):
        """construct rag chain and answer question by streaming"""

        # llm
        chat = CustomChatModel(api_key=self.api_key, base_url=self.base_url, model=self.model, temperature = self.temperature)
        # retriever
        retriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={'k': self.top_k})  # 默认similarity，k=4

        # prompt template
        prompt = ChatPromptTemplate.from_template(self.template)

        # chain
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | chat
        )
        for chunk in rag_chain.stream(question):
            yield chunk.content

if __name__ == '__main__':
    vectordb = indexing()

    rag_chain1 = RAGChain(model="gpt-3.5-turbo", top_k=5, vectordb=vectordb, provider="zhizengzeng",
                                  api_key=API_KEY, base_url=BASE_URL)
    response = rag_chain1.answer("介绍一下欧提努斯")
    print(response)

    rag_chain = RAGChain(model="gpt-3.5-turbo", top_k=5, vectordb=vectordb, provider="zhizengzeng",
                                 api_key=API_KEY, base_url=BASE_URL)
    for chunk in rag_chain.answer_stream("介绍一下欧提努斯"):
        print(chunk, end="|", flush=True)