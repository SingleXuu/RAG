from typing import List, Optional
from xml.dom.expatbuilder import TEXT_NODE

from elasticsearch import Elasticsearch
from langchain_core.retrievers import BaseRetriever
from llama_index.core import QueryBundle
from llama_index.core.schema import QueryType, NodeWithScore
from pydantic import Field

from preprocess.get_text_id_mapping import text_node_id_mapping

url = f"http://localhost:9200"

class CustomBM25Retriever(BaseRetriever):
    # es_client = Elasticsearch("http://localhost:9200")
    es_client: Optional[Elasticsearch] = Field(
        default=None,
        exclude=True  # 不参与序列化
    )
    top_k: int  # 添加类型注解

    def __init__(self,top_k):
        super().__init__(top_k=top_k)
        self.es_client = Elasticsearch([{'host':'localhost','port':9200,'scheme':'http'}])

    def _get_relevant_documents(self,query:QueryType) -> List[NodeWithScore]:
        if isinstance(query,str):
            query = QueryBundle(query)
        else:
            query = query

        result = []
        dsl = {
            'query':{
                'match':{
                    'content': query.query_str
                }
            },
            "size": self.top_k
        }
        search_result = self.es_client.search(index='docs',body=dsl)
        if search_result['hits']['hits']:
            for record in search_result['hits']['hits']:
                text = record['_source']['content']
                node_with_score = NodeWithScore(
                    node=TEXT_NODE(
                        text=text,
                        id_=text_node_id_mapping[text],
                        score=record['_score']
                    )
                )
                result.append(node_with_score)

if __name__ == '__main__':
    from pprint import pprint
    custom_bm25_retriever = CustomBM25Retriever(top_k=3)
    query = "美日半导体协议是由哪两部门签署的？美日半导体协议是由美国商务部和日本经济产业省签署的。"
    t_result = custom_bm25_retriever.get_relevant_documents(query=query)
    pprint(t_result)