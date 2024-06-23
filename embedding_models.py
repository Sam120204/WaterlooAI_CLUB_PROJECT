# embedding_models.py
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import json
import time

def get_bert_embeddings(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Use the mean pooling of token embeddings
    return embeddings.detach().numpy().tolist()  # Convert tensor to list

def get_biobert_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Use the mean pooling of token embeddings
    return embeddings.detach().numpy().tolist()  # Convert tensor to list

if __name__ == "__main__":
    # Load the data from pubmed_data.json
    with open('pubmed_data.json', 'r') as file:
        articles = json.load(file)
    
    # Generate embeddings for each article summary using BERT
    bert_embeddings = [get_bert_embeddings(article['summary']) for article in articles]
    
    # Generate embeddings for each article summary using BioBERT
    biobert_embeddings = [get_biobert_embeddings(article['summary']) for article in articles]
    
    # Save embeddings to JSON files
    with open('bert_embeddings.json', 'w') as file:
        json.dump(bert_embeddings, file)
    
    with open('biobert_embeddings.json', 'w') as file:
        json.dump(biobert_embeddings, file)
    
    print(f"Generated embeddings for {len(articles)} articles using BERT.")
    print(f"Generated embeddings for {len(articles)} articles using BioBERT.")
