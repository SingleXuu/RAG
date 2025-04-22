from config import Config

from langchain.vectorstores import ElasticVectorSearch
from langchain_community.vectorstores import ElasticsearchStore
from langchain_community.vectorstores.elasticsearch import ApproxRetrievalStrategy
from torch.backends.opt_einsum import strategy


def setup_vectordb(hf,index_name):
    with open('simple.cfg') as f:
        cfg = Config(f)

    # url = f"http://ragtest-4s4.public.cn-hangzhou.es-serverless.aliyuncs.com:9200"
    url = f"http://localhost:9200"
    # return ElasticsearchStore(
    #     embedding=hf,
    #     es_url=url,
    #     index_name=index_name,
    #     strategy=ApproxRetrievalStrategy(hybrid=True, rrf=True),
    # )
    return ElasticVectorSearch(
        embedding=hf,
        elasticsearch_url= url,
        index_name=index_name
    )

