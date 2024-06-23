# chromadb_store.py
import chromadb
import json

def flatten_embeddings(embeddings):
    # Ensure each embedding is a flat list
    return embeddings  # Assuming embeddings are already in the correct format

def store_in_chromadb(articles, embeddings):
    # Initialize ChromaDB client
    client = chromadb.HttpClient(host="localhost", port=8000)
    
    # Create a collection
    collection_name = "pubmed_articles"
    try:
        client.create_collection(name=collection_name)
    except Exception as e:
        print(f"Collection creation failed: {e}")
    
    # Prepare data for insertion
    ids = [str(i) for i in range(1, len(articles) + 1)]
    documents = [article['summary'] for article in articles]
    metadatas = [{'title': article['title']} for article in articles]
    
    # Insert data into the collection
    collection = client.get_collection(collection_name)
    collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
    
    # Verify insertion
    print("\nInserted documents:")
    for doc_id in ids:
        doc = collection.get(ids=[doc_id])
        print(doc)

if __name__ == "__main__":
    # Load the data from pubmed_data.json
    with open('pubmed_data.json', 'r') as file:
        articles = json.load(file)
    
    # Load the embeddings from embeddings.json
    with open('embeddings.json', 'r') as file:
        embeddings = json.load(file)
    
    # Store articles and embeddings in ChromaDB
    store_in_chromadb(articles, embeddings)
    
    print("Data and embeddings stored in ChromaDB.")
