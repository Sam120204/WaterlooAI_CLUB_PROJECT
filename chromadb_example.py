import chromadb

# Initialize the client
client = chromadb.HttpClient(host="localhost", port=8000)

# Create a collection
collection_name = "example_collection"
client.create_collection(name=collection_name)

# Prepare data
ids = ["1", "2"]
documents = ["This is document 1", "This is document 2"]
metadatas = [{"title": "Sample 1"}, {"title": "Sample 2"}]

# Print prepared data for verification
print("Prepared data:")
print("IDs:", ids)
print("Documents:", documents)
print("Metadatas:", metadatas)

# Insert data
collection = client.get_collection(collection_name)
collection.add(documents=documents, metadatas=metadatas, ids=ids)

# Verify insertion
print("\nInserted documents:")
for doc_id in ids:
    doc = collection.get(ids=[doc_id])
    print(doc)

# Query data
query_text = "This is document 1"
results = collection.query(query_texts=[query_text], n_results=2)

# Print results
print("\nQuery Results:")
print(results)
for result in results['documents'][0]:
    print(result)
