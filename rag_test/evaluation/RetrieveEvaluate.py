import json
from operator import index

import pandas as pd

from llama_index.core.schema import TextNode
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

def generator_dataset():
    # llm = ChatTongyi(model='qwen-long-latest',top_p=0.8,temperature=0.1,api_key='sk-e51e6fca8bb84f58b5e4bc5cb7216e15')
    llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX_LONGCONTEXT,api_key='sk-e51e6fca8bb84f58b5e4bc5cb7216e15')
    # Prompt to generate questions
    qa_generate_prompt_tmpl = """\
    Context information is below.
    
    ---------------------
    {context_str}
    ---------------------
    
    Given the context information and not prior knowledge.
    generate only questions based on the below query.
    
    You are a university professor. Your task is to set {num_questions_per_chunk} questions for the upcoming Chinese quiz.
    Questions throughout the test should be diverse. Questions should not contain options or start with Q1/Q2.
    Questions must be written in Chinese. The expression must be concise and clear. 
    It should not exceed 15 Chinese characters. Words such as "这", "那", "根据", "依据" and other punctuation marks 
    should not be used. Abbreviations may be used for titles and professional terms.
    """
    ##生成问题
    nodes = []
    data_df = pd.read_csv("../data/doc_qa_dataset.csv",encoding="utf-8")
    for i ,row in data_df.iterrows():
        if len(row["content"]) > 80 and i > 96:
            node = TextNode(text=row["content"])
            node.id_ = f"node_{i+1}"
            nodes.append(node)

    doc_qa_dataset = generate_question_context_pairs(nodes,llm=llm,num_questions_per_chunk=1,qa_generate_prompt_tmpl=qa_generate_prompt_tmpl)
    with open("../data/doc_qa_dataset_test.json", "w", encoding="utf8") as f:
        json.dump(doc_qa_dataset.model_dump(), f, indent=4, ensure_ascii=False)


def setup_retriever():
    retriever = index.as_retriever()
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever
    )

    retriever_evaluator.evaluate(
        query="query", expected_ids=["node_id1", "node_id2"]
    )


if __name__ == "__main__":
    generator_dataset()

