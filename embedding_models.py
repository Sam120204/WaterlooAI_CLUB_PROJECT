from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import json

def get_bert_embeddings(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Use the mean pooling of token embeddings
    return embeddings.detach().numpy().tolist()  # Convert tensor to list

def get_biobert_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Use the mean pooling of token embeddings
    return embeddings.detach().numpy().tolist()  # Convert tensor to list

if __name__ == "__main__":
    # Load the data from pubmed_data.json
    with open('D:/桌面/CS Waterloo/WAT_CLUB/pubmed_data.json', 'r') as file:
        articles = json.load(file)
    
    # Generate embeddings for each article summary using BERT
    bert_embeddings = []
    for i, article in enumerate(articles):
        embedding = get_bert_embeddings(article['summary'])
        bert_embeddings.append(embedding)
        print(f"Generated BERT embedding for article {i+1}/{len(articles)}")

    # Generate embeddings for each article summary using BioBERT
    biobert_embeddings = []
    for i, article in enumerate(articles):
        embedding = get_biobert_embeddings(article['summary'])
        biobert_embeddings.append(embedding)
        print(f"Generated BioBERT embedding for article {i+1}/{len(articles)}")
    
    # Save embeddings to JSON files
    with open('D:/桌面/CS Waterloo/WAT_CLUB/bert_embeddings.json', 'w') as file:
        json.dump(bert_embeddings, file)
    
    with open('D:/桌面/CS Waterloo/WAT_CLUB/biobert_embeddings.json', 'w') as file:
        json.dump(biobert_embeddings, file)
    
    print(f"Generated embeddings for {len(articles)} articles using BERT.")
    print(f"Generated embeddings for {len(articles)} articles using BioBERT.")
