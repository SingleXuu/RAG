from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

## for vector store
from langchain.vectorstores import ElasticVectorSearch
from elasticsearch import Elasticsearch

url = f"http://localhost:9200"

def parse_book(filepath):
    loader = TextLoader(filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs

def parse_triplets(filepath):
    docs = parse_book(filepath)
    result = []
    for i in range(len(docs) - 2):
        concat_str = docs[i].page_content + " " + docs[i+1].page_content + " " + docs[i+2].page_content
        result.append(concat_str)
    return result


# def loadBookTriplets(filepath, url, hf, db, index_name):
#     with open('simple.cfg') as f:
#         cfg = Config(f)
#
#     fingerprint = cfg['ES_FINGERPRINT']
#     es = Elasticsearch([url],http_compress=True)
#
#     ## Parse the book if necessary
#     if not es.indices.exists(index=index_name):
#         print(f'\tThe index: {index_name} does not exist')
#         print(">> 1. Chunk up the Source document")
#
#         results = parse_triplets(filepath)
#
#         print(">> 2. Index the chunks into Elasticsearch")
#
#         elastic_vector_search = ElasticVectorSearch.from_documents(docs,
#                                                                    embedding=hf,
#                                                                    elasticsearch_url=url,
#                                                                    index_name=index_name)
#     else:
#         print("\tLooks like the pdfs are already loaded, let's move on")


def loadBookBig(filepath, url, hf, db, index_name):
    es = Elasticsearch([url],http_compress=True)

    ## Parse the book if necessary
    if not es.indices.exists(index=index_name):
        print(f'\tThe index: {index_name} does not exist')
        print(">> 1. Chunk up the Source document")

        docs = parse_book(filepath)

        # print(docs)

        print(">> 2. Index the chunks into Elasticsearch")

        elastic_vector_search = ElasticVectorSearch.from_documents(docs,
                                                                   embedding=hf,
                                                                   elasticsearch_url=url,
                                                                   index_name=index_name)
    else:
        print("\tLooks like the pdfs are already loaded, let's move on")


def loadPdfChunks(chunks, url, hf, db, index_name):
    es = Elasticsearch([url],http_compress=True)

    ## Parse the book if necessary
    if not es.indices.exists(index=index_name):
        print(f'\tThe index: {index_name} does not exist')
        print(">> 2. Index the chunks into Elasticsearch")

        print("url: ", url)
        print("index_name", index_name)

        elastic_vector_search = db.from_texts(chunks,
                                              embedding=hf,
                                              elasticsearch_url=url,
                                              index_name=index_name
                                              )
    else:
        print("\tLooks like the pdfs are already loaded, let's move on")


