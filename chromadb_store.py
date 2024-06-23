import numpy as np
import chromadb

# Load embeddings from the .npy file
embeddings = np.load('biobert_embeddings.npy')

# Ensure embeddings are in the correct shape (number of embeddings, embedding dimension)
print(f'Shape of embeddings: {embeddings.shape}')

# Check the first embedding to confirm it is being loaded correctly
print(f'First embedding: {embeddings[0]}')

# Create document IDs and prepare documents for upsertion
documents = [
    {
        "id": str(i),  # Ensure the ID is a string
        "embedding": embeddings[i].tolist(),  # Convert numpy array to list
        "metadata": {
            "document_id": i
        }
    }
    for i in range(len(embeddings))
]

# Print the first document to ensure it is correctly formatted
print(f'First document: {documents[0]}')

# Initialize ChromaDB client and collection
client = chromadb.Client()
collection = client.get_or_create_collection(name='biobert_collection')

# Split documents into separate lists for IDs, embeddings, and metadata
ids = [doc["id"] for doc in documents]
embeddings_list = [doc["embedding"] for doc in documents]
metadata_list = [doc["metadata"] for doc in documents]

# Upsert the documents into the collection
try:
    collection.upsert(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadata_list
    )
    print("Documents have been upserted successfully.")
except ValueError as e:
    print(f"ValueError: {e}")
    # Print document IDs to help debug the issue
    for doc in documents:
        print(f"ID: {doc['id']} - Type: {type(doc['id'])}")

