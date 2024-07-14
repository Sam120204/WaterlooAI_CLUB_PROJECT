import json
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_response(retrieved_docs, query):
    context = " ".join([doc['content'] for doc in retrieved_docs['documents']])
    prompt = f"{query}\nContext: {context}\nAnswer:"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Ensure this is the correct model you have access to
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    return response.choices[0].message['content']

if __name__ == "__main__":
    with open('processed_pubmed_data.json', 'r') as f:
        articles = json.load(f)
    
    documents = [chunk for article in articles for chunk in article['chunks']]
    
    # Assuming best_result contains the best retrieved documents from verify_retriever.py
    best_result = {
        'documents': [
            {'content': 'Example content 1'},
            {'content': 'Example content 2'},
            # Add more document contents as needed
        ]
    }
    
    query = "machine learning applications in healthcare"
    response = generate_response(best_result, query)
    print(response)
