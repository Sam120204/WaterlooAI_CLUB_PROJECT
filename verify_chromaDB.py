# verify_chromaDB.py
import chromadb
import json

# Initialize ChromaDB client
client = chromadb.HttpClient(host="localhost", port=8000)

# Access the collection
collection_name = "pubmed_articles"
collection = client.get_collection(collection_name)

# Fetch stored documents by IDs (if known) or fetch a subset of documents
# Here, we assume you have document IDs stored or you can iterate over a range

try:
    # Example: Fetch first 10 documents by their IDs
    ids = [str(i) for i in range(1, 11)]
    for doc_id in ids:
        doc = collection.get(ids=[doc_id])
        print(f"Document ID: {doc_id}, Data: {doc}")
        print(f"Embeddings: {doc['embeddings']}")  # Check if embeddings are present

except Exception as e:
    print(f"Error fetching documents: {e}")
