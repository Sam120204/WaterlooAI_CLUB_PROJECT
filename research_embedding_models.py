from transformers import BertTokenizer, BertModel
import torch

def get_bert_embeddings(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state
    return embeddings

if __name__ == "__main__":
    sample_text = "Hello, how are you?"
    embeddings = get_bert_embeddings(sample_text)
    
    print(embeddings)
