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

## Running the Scripts

### 1. Fetching Data from PubMed

The `fetch_pubmed_data.py` script scrapes articles from PubMed based on a specified query.

**Running the Script:**

```sh
python fetch_pubmed_data.py
```

**Output:**

- The script creates a file named `pubmed_data.json` containing the scraped articles.
- You will see a message in the terminal indicating the number of articles fetched.

### 2. Researching Embedding Models

The `embedding_models.py` script generates text embeddings using BERT.

**Running the Script:**

```sh
python embedding_models.py
```

**Output:**

- The script prints the embeddings for the sample text in the terminal.
- Ensure you see a tensor output indicating the embeddings were successfully generated.
- The script creates a file named `embeddings.json` containing the generated embeddings.

### 3. Using ChromaDB for Vector Storage

The `chromadb_store.py` script demonstrates how to insert and query data in ChromaDB.

**Running the Script:**

```sh
python chromadb_store.py
```

**Output:**

- The script prints the results of the insertion and query in the terminal.
- The output shows the stored documents with their embeddings and metadata.

### 4. Checking ChromaDB Endpoints

The `check_endpoints.py` script checks the status of the ChromaDB server endpoints.

**Running the Script:**

```sh
python check_endpoints.py
```

**Output:**

- The script prints the status codes for each endpoint, indicating whether they are accessible.

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
