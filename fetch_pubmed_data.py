# Web Scraping/Downloading and Storing Datasets
import requests
from bs4 import BeautifulSoup
import json

def fetch_pubmed(query, max_results=100):
    url = f"https://pubmed.ncbi.nlm.nih.gov/?term={query}&size={max_results}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('article', class_='full-docsum')
    
    results = []
    for article in articles:
        title = article.find('a', class_='docsum-title').text.strip()
        summary = article.find('div', class_='full-view-snippet').text.strip()
        results.append({'title': title, 'summary': summary})
    
    return results

if __name__ == "__main__":
    query = "COVID-19"
    articles = fetch_pubmed(query)
    
    # Save data to a JSON file
    with open('pubmed_data.json', 'w') as file:
        json.dump(articles, file)
    
    print(f"Fetched {len(articles)} articles.")
