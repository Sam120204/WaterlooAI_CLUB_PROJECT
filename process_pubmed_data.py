import json
import os

def chunk_text(text, chunk_size=512, overlap=128):
    """Chunks the text into smaller segments with optional overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    # Load data from pubmed_data.json
    with open('pubmed_data.json', 'r') as f:
        articles = json.load(f)

    # Process articles: chunk the text
    for article in articles:
        article['chunks'] = chunk_text(article['summary'])

    # Save the processed data to a new JSON file
    with open('processed_pubmed_data.json', 'w') as f:
        json.dump(articles, f)

    # Print the number of articles and a sample of chunks
    print(f"Number of articles processed: {len(articles)}")
    print("Sample chunks from the first article:")
    print(articles[0]['chunks'])
