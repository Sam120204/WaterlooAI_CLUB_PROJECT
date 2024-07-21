# langchain_pipeline.py
from langchain.agents import initialize_agent, AgentType
from langchain.chains import RunnableSequence
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import json
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import DocumentSearchPipeline
from dotenv import load_dotenv
import os
from Bio import Entrez

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

class MedicalMLPipeline:
    def __init__(self):
        self.entrez_email = "your_email@example.com"
        self.topic = "machine learning in healthcare"
        self.model_name = 'dmis-lab/biobert-base-cased-v1.1'
        self.query = "machine learning applications in healthcare"
        self.retriever = None
        self.document_store = None

    def fetch_pubmed_data(self, max_results=100):
        Entrez.email = self.entrez_email
        search_handle = Entrez.esearch(db="pubmed", term=self.topic, retmax=max_results)
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
        
        with open('pubmed_data.json', 'w') as f:
            json.dump(articles, f)
        return articles

    def chunk_text(self, text, chunk_size=512, overlap=128):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def process_articles(self):
        with open('pubmed_data.json', 'r') as f:
            articles = json.load(f)
        for article in articles:
            article['chunks'] = self.chunk_text(article['summary'])
        with open('processed_pubmed_data.json', 'w') as f:
            json.dump(articles, f)
        return articles

    def get_biobert_embeddings(self, texts):
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = BertModel.from_pretrained(self.model_name)
        model.eval()
        
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return np.array(embeddings)

    def generate_embeddings(self):
        with open('processed_pubmed_data.json', 'r') as f:
            data = json.load(f)
        articles = [f"{article['title']} {article.get('summary', '')}" for article in data]
        biobert_embeddings = self.get_biobert_embeddings(articles)
        np.save("biobert_embeddings.npy", biobert_embeddings)
        return biobert_embeddings

    def verify_retriever(self):
        with open('processed_pubmed_data.json', 'r') as f:
            articles = json.load(f)
        documents = [chunk for article in articles for chunk in article['chunks']]
        self.document_store = InMemoryDocumentStore()
        self.document_store.write_documents([{"content": doc} for doc in documents])
        
        self.retriever = DensePassageRetriever(
            document_store=self.document_store, 
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base", 
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base", 
            use_gpu=False
        )
        
        self.document_store.update_embeddings(self.retriever)
        search_pipeline = DocumentSearchPipeline(self.retriever)
        
        retrieved_docs = search_pipeline.run(query=self.query, params={"Retriever": {"top_k": 10}})
        return retrieved_docs

    def generate_response(self, retrieved_docs):
        # Access document content properly
        context = " ".join([doc.content for doc in retrieved_docs['documents']])
        prompt = f"{self.query}\nContext: {context}\nAnswer:"

        # Use LangChain's OpenAI LLM for response generation
        llm = ChatOpenAI(model_name="gpt-4", api_key=openai_api_key)
        prompt_template = PromptTemplate(input_variables=["context", "query"], template="{query}\nContext: {context}\nAnswer:")
        chain = RunnableSequence([prompt_template, llm])

        response = chain.invoke({"context": context, "query": self.query})
        return response

    def run_pipeline(self):
        self.fetch_pubmed_data()
        self.process_articles()
        self.generate_embeddings()
        retrieved_docs = self.verify_retriever()
        response = self.generate_response(retrieved_docs)
        print(response)

if __name__ == "__main__":
    pipeline = MedicalMLPipeline()
    pipeline.run_pipeline()
