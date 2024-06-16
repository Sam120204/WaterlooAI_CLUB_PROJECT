# Medical Data Processing and Research

This repository contains scripts for web scraping medical data, researching embedding models, and using a vector database for storing and querying embeddings. The primary focus is on utilizing PubMed and Cochrane as data sources and ChromaDB as the vector database.

## Contents

- `fetch_pubmed_data.py`: Script for scraping data from PubMed.
- `research_embedding_models.py`: Script for generating embeddings using BERT.
- `chromadb_example.py`: Script for inserting and querying data in ChromaDB.
- `requirements.txt`: File listing all dependencies.
- `research_notes.txt`: Placeholder for personal research notes and observations.

## Setup Instructions

### Prerequisites

- Python 3.x installed on your system.
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

The `research_embedding_models.py` script generates text embeddings using BERT.

**Running the Script:**

```sh
python research_embedding_models.py
```

**Output:**

- The script prints the embeddings for the sample text in the terminal.
- Ensure you see a tensor output indicating the embeddings were successfully generated.

### 3. Using ChromaDB for Vector Storage

The `chromadb_example.py` script demonstrates how to insert and query data in ChromaDB.

**Running the Script:**

```sh
python chromadb_example.py
```

**Output:**

- The script prints the results of the query in the terminal.
- The output shows the queried documents with their embeddings and metadata.

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
