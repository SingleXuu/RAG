from langchain_community.chat_models import ChatTongyi
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

def make_the_llm():
    llm = ChatTongyi(model='qwen-long-latest',top_p=0.8,temperature=0.1,api_key='sk-e51e6fca8bb84f58b5e4bc5cb7216e15')

    template_informed = """
    I know: {context}
    when asked: {question}
    my response is: """

    prompt_informed = PromptTemplate(template=template_informed, input_variables=["context", "question"])

    return LLMChain(prompt=prompt_informed, llm=llm)