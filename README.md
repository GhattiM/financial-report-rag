# Financial Documents Analysis RAG System

A RAG (Retrieval-Augmented Generation) system designed for analyzing and comparing SEC 10-K financial reports across multiple companies. This system uses a hybrid retrieval pipeline (BM25 + Vector Search) and advanced reranking to provide accurate financial insights.

## Features
- **Smart Ingestion:** Automated PDF extraction and chunking optimized for financial reports.
- **Hybrid Retrieval:** Combines semantic vector search with keyword-based BM25 search.
- **Advanced Reranking:** Integrates Cohere's Rerank v3.5 via OpenRouter for high-precision retrieval.
- **Context Optimization:** Utilizes `LongContextReorder` to mitigate "Lost in the Middle" context issues.
- **Evaluation:** Integrated Ragas evaluation pipeline for measuring faithfulness, relevancy, and correctness.
- **Interactive UI:** Streamlit interface for easy document analysis.

## Tech Stack
- **Language:** Python 3.13
- **RAG Framework:** LangChain
- **Vector Database:** ChromaDB (local)
- **Embeddings:** HuggingFace `BAAI/bge-base-en-v1.5`
- **LLM:** `google/gemini-3.1-flash-lite-preview` (via OpenRouter)
- **Reranker:** `cohere/rerank-4-pro` (via OpenRouter)
- **PDF Extraction:** `pdfplumber`
- **Evaluation:** `Ragas`

## Project Structure
- `src/ingest.py`: PDF loading, text splitting, and vector store creation.
- `src/query.py`: Hybrid retrieval pipeline and RAG chain implementation.
- `src/evaluate.py`: Ragas evaluation script with custom JSON cleaning.
- `src/generate_questions.py`: Utility to generate evaluation datasets from 10-K data.
- `src/app.py`: Streamlit application.
- `data/10k/`: Directory for source PDF filings.
- `data/golden_dataset.json`: Curated evaluation dataset.

## Setup Instructions

### 1. Prerequisites
- Python 3.13
- OpenRouter API Key
- Cohere API Key (via OpenRouter)

### 2. Installation
Clone the repository and install dependencies:
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory and add your API keys:
```env
OPENROUTER_API_KEY=your_openrouter_api_key
COHERE_API_KEY=your_cohere_api_key
```

### 4. GitHub Actions (Optional)
This project includes a PR evaluation workflow. To use it:
1.  Go to your GitHub repository settings.
2.  Navigate to **Secrets and variables > Actions**.
3.  Add the following **Repository secrets**:
    - `OPENROUTER_API_KEY`: Your OpenRouter API key.
    - `COHERE_API_KEY`: Your Cohere API key.

The workflow will automatically run the Ragas evaluation on every pull request and post a summary as a comment.

## Usage

### Ingest Data
Place your SEC 10-K PDFs in `data/10k/` and run:
```bash
python src/ingest.py
```

### Run Interactive Terminal
Start the CLI-based query terminal:
```bash
python src/query.py
```

### Run Streamlit UI
Launch the web interface:
```bash
streamlit run src/app.py
```

### Run Evaluation
Generate Ragas performance metrics:
```bash
python src/evaluate.py
```

## Evaluation Results
The system is evaluated using the following Ragas metrics:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
- Answer Correctness

Results are saved to `data/eval_ragas_results.json`.

## License
MIT License (or specify your license)
