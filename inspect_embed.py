import json

# Load embeddings
with open('embeddings.json', 'r') as file:
    embeddings = json.load(file)

# Load articles
with open('pubmed_data.json', 'r') as file:
    articles = json.load(file)

# Check if embeddings and articles match
for i, article in enumerate(articles):
    print(f"Article {i+1}: {article['title']}")
    print(f"Embedding {i+1}: {embeddings[i][:10]}...")  # Print first 10 values of the embedding for brevity

print(f"Total articles: {len(articles)}")
print(f"Total embeddings: {len(embeddings)}")
