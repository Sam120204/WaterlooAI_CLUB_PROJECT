from transformers import BertTokenizer, BertModel
import torch
import json
import time

def get_bert_embeddings(text):
    # Load tokenizer and model with retries and logging
    retries = 3
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1} to load the tokenizer and model...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            print("Model and tokenizer loaded successfully.")
            break
        except Exception as e:
            print(f"Error loading model: {e}")
            if attempt < retries - 1:
                print("Retrying...")
                time.sleep(5)
            else:
                print("Failed to load model after several attempts. Exiting.")
                raise

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Use the mean pooling of token embeddings
    return embeddings.detach().numpy().tolist()  # Convert tensor to list

if __name__ == "__main__":
    # Load the data from pubmed_data.json
    with open('pubmed_data.json', 'r') as file:
        articles = json.load(file)
    
    # Generate embeddings for each article summary
    embeddings = [get_bert_embeddings(article['summary']) for article in articles]
    
    # Save embeddings to a JSON file
    with open('embeddings.json', 'w') as file:
        json.dump(embeddings, file)
    
    print(f"Generated embeddings for {len(embeddings)} articles.")
