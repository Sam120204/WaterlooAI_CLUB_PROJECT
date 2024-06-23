# Medical Data Processing and Research

This repository contains scripts for web scraping medical data, researching embedding models, and using a vector database for storing and querying embeddings. The primary focus is on utilizing PubMed and Cochrane as data sources and ChromaDB as the vector database.

## Contents

- `fetch_pubmed_data.py`: Script for scraping data from PubMed.
- `embedding_models.py`: Script for generating embeddings using BERT.
- `chromadb_store.py`: Script for inserting and querying data in ChromaDB.
- `check_endpoints.py`: Script for checking the status of ChromaDB endpoints.
- `requirements.txt`: File listing all dependencies.
- `research_notes.txt`: Placeholder for personal research notes and observations.
- `pubmed_data.json`: JSON file containing scraped articles from PubMed.
- `embeddings.json`: JSON file containing generated embeddings for the articles.

## Setup Instructions

### Prerequisites

- Python 3.x installed on your system.
- Docker installed on your system.
- Internet connection for installing dependencies and fetching data.

### Creating a Virtual Environment (Optional but Recommended)

1. Create a virtual environment:
   ```sh
   python -m venv myenv
   ```
2. Activate the virtual environment:
   - On macOS/Linux:
     ```sh
     source myenv/bin/activate
     ```
   - On Windows:
     ```sh
     myenv\Scripts\activate
     ```

### Installing Dependencies

Install the necessary libraries using the provided `requirements.txt` file:

```sh
pip install -r requirements.txt
```

### Setting Up and Running ChromaDB Server with Docker

1. **Pull the ChromaDB Docker image and run the container:**
   ```sh
   docker run -d -p 8000:8000 chromadb/chroma
   ```

2. **Verify the Docker container is running:**
   ```sh
   docker ps
   ```

   You should see a container running with the `chromadb/chroma` image.

3. **Check the logs to ensure the server is running correctly:**
   ```sh
   docker logs <container_id>
   ```

   Replace `<container_id>` with the ID of the ChromaDB container listed in the output of the `docker ps` command.

To ensure a smooth workflow, here’s the order in which you should run the scripts:

1. **`fetch_pubmed_data.py`**: Fetch the articles and save them to `pubmed_data.json`.
2. **`embedding_models.py`**: Generate embeddings for the articles using both BERT and BioBERT and save them to `bert_embeddings.json` and `biobert_embeddings.json`.
3. **`chromadb_store.py`**: Store the articles and their embeddings in ChromaDB.
4. **`verify_chromaDB.py`**: Verify that the articles and embeddings have been correctly stored in ChromaDB.

### Step-by-Step Instructions:

1. **Fetch Articles**:
    - Run `fetch_pubmed_data.py` to fetch articles from PubMed and save them to `pubmed_data.json`.

    ```sh
    python fetch_pubmed_data.py
    ```

2. **Generate Embeddings**:
    - Run `embedding_models.py` to generate embeddings using both BERT and BioBERT, and save them to `bert_embeddings.json` and `biobert_embeddings.json`.

    ```sh
    python embedding_models.py
    ```

3. **Store Data in ChromaDB**:
    - Run `chromadb_store.py` to store the articles and their embeddings in ChromaDB.

    ```sh
    python chromadb_store.py
    ```

4. **Verify Stored Data**:
    - Run `verify_chromaDB.py` to verify that the articles and embeddings are correctly stored in ChromaDB.

    ```sh
    python verify_chromaDB.py
    ```


## Research and Collaboration

### Reading and Contributing to the Research Document

- Access the shared research document to review existing entries and datasets.
- Note any interesting datasets or potential collaboration opportunities in your `research_notes.txt`.

### Adding Personal Notes

- Open a text editor and create or update the `research_notes.txt` file with your observations and potential collaboration ideas.
- Save the file with your inputs for future reference.

## Additional Notes

- Ensure all scripts are run from the project directory where the `requirements.txt` file is located.
- If you encounter any issues, check for error messages and ensure all required libraries are installed correctly.
- If you have trouble accessing the ChromaDB server, ensure the Docker container is running and check the logs for any errors.
