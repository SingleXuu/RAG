from dotenv import load_dotenv
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models.litellm_router import model_extra_key_name

from langchain_ollama.llms import OllamaLLM

from langchain_community.chat_models import ChatTongyi
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def prepare_data():
    loader = WebBaseLoader("https://baike.baidu.com/item/AIGC?\
    fromModule = lemma_search - box")

    documents=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(chunks[0].page_content)
    return chunks

def embedding_data(chunks):
    rag_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vector_store = Chroma.from_documents(documents=chunks,
embedding=rag_embeddings,persist_directory="./chroma_langchain_db")
    retriever = vector_store.as_retriever()
    return vector_store,retriever

#使用ollama服务
llm = OllamaLLM(model="qwen2:7b-instruct-q4_0")
template = """您是问答任务的助理。
使用以下检索到的上下文来回答问题。
如果你不知道答案，就说你不知道。
最多使用三句话，不超过100字，保持答案简洁。
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chunks = prepare_data()
vector_store,retriever = embedding_data(chunks)

def generate_answer(question):
    llm = ChatTongyi(model='qwen-long',top_p=0.8,temperature=0.1,api_key='sk-e51e6fca8bb84f58b5e4bc5cb7216e15')
    rag_chain=(
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    resp = rag_chain.invoke(question)
    print(resp)



if __name__ == "__main__":
    query = "艾伦•图灵的论文叫什么"
    generate_answer(query)