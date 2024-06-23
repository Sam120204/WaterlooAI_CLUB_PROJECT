import chromadb
import numpy as np
import json

# Load articles and embeddings
with open("pubmed_data.json", "r") as file:
    data = json.load(file)

# Extract article text (using title and other fields as necessary)
articles = [f"{article['title']} {article.get('abstract', '')}" for article in data]

# Load embeddings
bert_embeddings = np.load("bert_embeddings.npy")

# Ensure embeddings dimensions match with your model
print(f"BERT Embeddings shape: {bert_embeddings.shape}")
assert len(articles) == bert_embeddings.shape[0], "Mismatch between number of articles and embeddings"

# Initialize ChromaDB client
client = chromadb.Client()

# Get or create a collection
collection = client.get_or_create_collection("pubmed_articles")

# Adding documents and embeddings to the collection
try:
    for idx, (article, embedding) in enumerate(zip(articles, bert_embeddings)):
        doc_id = str(idx)
        try:
            collection.add(
                documents=[article],
                embeddings=[embedding.tolist()],
                ids=[doc_id]
            )
            print(f"Added document ID {doc_id} to collection 'pubmed_articles'")
        except Exception as e:
            print(f"Error adding document ID {doc_id}: {e}")
except Exception as e:
    print(f"Error adding documents: {e}")

# Display collections after storing data
collections = client.list_collections()
print("Collections after storing data:")
for col in collections:
    print(f"- name='{col.name}' id={col.id}")

# Check if collection has been populated correctly
try:
    collection = client.get_collection("pubmed_articles")
    sample_data = collection.query(
        query_embeddings=[bert_embeddings[0].tolist()],
        n_results=5
    )
    print(f"Sample data from collection 'pubmed_articles': {sample_data}")
except Exception as e:
    print(f"Error querying collection: {e}")
