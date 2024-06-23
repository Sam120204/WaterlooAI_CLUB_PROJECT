import json
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def get_bert_embeddings(texts):
    embeddings = []
    with torch.no_grad():
        for text in texts:
            # Encode text
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            # Get the mean pooling of the token embeddings
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

if __name__ == "__main__":
    # Load data
    with open('pubmed_data.json', 'r') as f:
        data = json.load(f)

    articles = [f"{article['title']} {article['summary']}" for article in data]

    # Generate BERT embeddings
    bert_embeddings = get_bert_embeddings(articles)

    # Save embeddings to a file
    np.save("bert_embeddings.npy", bert_embeddings)
