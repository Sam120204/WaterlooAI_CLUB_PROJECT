# import json
# import numpy as np
# from transformers import BertTokenizer, BertModel
# import torch
# from Bio import Entrez

# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Load pre-trained model (weights)
# model = BertModel.from_pretrained('bert-base-uncased')
# model.eval()


# def fetch_pubmed_data(topic, max_results=100):
#     Entrez.email = "zhongjiayou1202@gmail.com"  # Always provide your email address when using Entrez
#     search_handle = Entrez.esearch(db="pubmed", term=topic, retmax=max_results)
#     search_results = Entrez.read(search_handle)
#     search_handle.close()
#     id_list = search_results["IdList"]
    
#     fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="xml")
#     fetched_data = Entrez.read(fetch_handle)
#     fetch_handle.close()
    
#     articles = []
#     for article in fetched_data["PubmedArticle"]:
#         title = article["MedlineCitation"]["Article"]["ArticleTitle"]
#         abstract = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [""])[0]
#         articles.append({"title": title, "summary": abstract})
    
#     return articles

# if __name__ == "__main__":
#     # Define the topic to search for in PubMed
#     topic = "machine learning in healthcare"
    
#     # Fetch data from PubMed
#     articles = fetch_pubmed_data(topic)
    
#     # Save data to a JSON file
#     with open('pubmed_data.json', 'w') as f:
#         json.dump(articles, f)
    
#     # Load data
#     with open('pubmed_data.json', 'r') as f:
#         data = json.load(f)

#     # Print the number of articles
#     print(f"Number of articles fetched: {len(data)}")

#     # Combine title and summary for each article
#     combined_texts = [f"{article['title']} {article.get('summary', '')}" for article in data]

import json
from Bio import Entrez

def fetch_pubmed_data(topic, max_results=100):
    Entrez.email = "zhongjiayou1202@gmail.com"  # Always provide your email address when using Entrez
    search_handle = Entrez.esearch(db="pubmed", term=topic, retmax=max_results)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    id_list = search_results["IdList"]
    
    fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="xml")
    fetched_data = Entrez.read(fetch_handle)
    fetch_handle.close()
    
    articles = []
    for article in fetched_data["PubmedArticle"]:
        title = article["MedlineCitation"]["Article"]["ArticleTitle"]
        abstract = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [""])[0]
        articles.append({"title": title, "summary": abstract})
    
    return articles

if __name__ == "__main__":
    # Define the topic to search for in PubMed
    topic = "machine learning in healthcare"
    
    # Fetch data from PubMed
    articles = fetch_pubmed_data(topic)
    
    # Save data to a JSON file
    with open('pubmed_data.json', 'w') as f:
        json.dump(articles, f)
    
    # Print the number of articles
    print(f"Number of articles fetched: {len(articles)}")
