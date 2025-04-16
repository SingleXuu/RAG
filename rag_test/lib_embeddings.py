from langchain_community.embeddings import HuggingFaceEmbeddings

def setup_embeddings():
    # Huggingface embedding setup
    print(">> Prep. Huggingface embedding setup")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    # return HuggingFaceEmbeddings(model_name=model_name)

    rag_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    return rag_embeddings