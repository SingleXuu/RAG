from http.client import responses

from datasets import load_dataset, Dataset
from langchain_community.chat_models import ChatTongyi
from pyarrow.dataset import dataset

from ragas import EvaluationDataset, SingleTurnSample, RunConfig
from ragas.llms import LangchainLLMWrapper

from ragas.metrics import LLMContextRecall, Faithfulness, faithfulness, FactualCorrectness, SemanticSimilarity,answer_relevancy,context_recall,context_precision
from ragas import evaluate
from sqlalchemy.testing import run_as_contextmanager
from streamlit import metric
from torch.nn.functional import embedding

import lib_llm
import lib_vectordb
from lib_embeddings import setup_embeddings
from myapp import handle_userinput

# dataset = load_dataset("explodinggradients/amnesty_qa","english_v3")
# samples = []
# for row in dataset['eval']:
#     print("response = ",row['response'])
#     sample = SingleTurnSample(
#         user_input=row['user_input'],
#         reference=row['reference'],
#         response=row['response'],
#         retrieved_contexts=row['retrieved_contexts']
#     )
#     samples.append(sample)
#
# eval_dataset = EvaluationDataset(samples=samples)
#
# evaluator_llm = LangchainLLMWrapper(ChatTongyi(model='qwen-long-latest',top_p=0.8,temperature=0.1,api_key='sk-e51e6fca8bb84f58b5e4bc5cb7216e15'))
# metrics = [LLMContextRecall(), FactualCorrectness(), Faithfulness()]
# results = evaluate(dataset=eval_dataset, metrics=metrics, llm=evaluator_llm)
# print(results)
# df = results.to_pandas()
# df.to_csv("ragas_evaluation_results.csv", index=False)
# print(df.head())

answers = []
context = []
questions = ["徐道明的联系方式是多少？","徐道明的邮箱是多少？","徐道明的年龄是多少？"]
ground_truths = ["徐道明的电话号码是18767176339","徐道明的邮箱是1293552247@qq.com","徐道明今年29岁"]

llm = LangchainLLMWrapper(ChatTongyi(model='qwen-long-latest',top_p=0.8,temperature=0.1,api_key='sk-e51e6fca8bb84f58b5e4bc5cb7216e15'))
embeddings = setup_embeddings()
llm_chain_informed = lib_llm.make_the_llm()
db = lib_vectordb.setup_vectordb(embeddings, "pdf_docs")

for query in questions:
    answers.append(handle_userinput(db, llm_chain_informed, query))
    context.append([docs.page_content for docs in db.similarity_search(query)])

# data = {
#     "question" : questions,
#     "answer" : answers,
#     "contexts" : context,
#     "ground_truth":ground_truths
# }
data = {
    "user_input" : questions,
    "response" : answers,
    "retrieved_contexts" : context,
    "reference":ground_truths
}
#         user_input=row['user_input'],
#         reference=row['reference'],
#         response=row['response'],
#         retrieved_contexts=row['retrieved_contexts']
dataset= Dataset.from_dict(data)
print(dataset)

run_config = RunConfig(max_retries=10,max_wait=60,log_tenacity=True)

result = evaluate(
    dataset = dataset,
    llm = llm,
    embeddings = embeddings,
    run_config=run_config,
    metrics=[
        faithfulness,answer_relevancy,context_recall,context_precision
    ]
)

print(result)
df = result.to_pandas()
df.to_csv("test.csv", index=False)
print(df)
