from config import Config

from langchain.vectorstores import ElasticVectorSearch

def setup_vectordb(hf,index_name):
    with open('simple.cfg') as f:
        cfg = Config(f)

    url = f"http://localhost:9200"

    return ElasticVectorSearch(
        embedding=hf,
        elasticsearch_url= url,
        index_name=index_name
    )

