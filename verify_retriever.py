import json
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import DocumentSearchPipeline

def verify_retriever(retriever, document_store, query):
    search_pipeline = DocumentSearchPipeline(retriever)

    # Parameters to experiment with
    top_k = [5, 10]
    best_result = None

    for k in top_k:
        retrieved_docs = search_pipeline.run(query=query, params={"Retriever": {"top_k": k}})
        if not best_result or len(retrieved_docs["documents"]) > len(best_result["documents"]):
            best_result = retrieved_docs
    
    return best_result

if __name__ == "__main__":
    with open('processed_pubmed_data.json', 'r') as f:
        articles = json.load(f)
    
    documents = [chunk for article in articles for chunk in article['chunks']]
    
    document_store = InMemoryDocumentStore()
    document_store.write_documents([{"content": doc} for doc in documents])
    
    retriever = DensePassageRetriever(document_store=document_store, query_embedding_model="facebook/dpr-question_encoder-single-nq-base", passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base", use_gpu=False)
    
    # Update embeddings for all documents in the document store
    document_store.update_embeddings(retriever)
    
    query = "machine learning applications in healthcare"
    best_result = verify_retriever(retriever, document_store, query)
    print(best_result)
