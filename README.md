# Biomedical RAG System

A modular, production-ready agentic Retrieval-Augmented Generation (RAG) system for biomedical literature search, synthesis, and citation. Built with LangChain, ChromaDB(Vector Store) and DeepSeek.

## Features
- Pubmed Source Retrieval
- Biomedical NER and query expansion
- RAG workflow
- Embeddings with Sentence Transformers
- Inline citations and references
- Modular, extensible, and robust codebase

## Installation

1. **Clone the repository:**
```bash
   git clone 
cd MedLitAssistant
```
2.**create conda environment:**
   ```bash
   conda create -n medlit 
   conda activate medlit
   ``` 

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Copy `env_template.txt` to `.env` and fill in your API keys (OpenAI, Semantic Scholar, etc.)

## Usage

- **Run the agentic RAG workflow:**
  ```chainlit run src/ragagent/pubmedrag_app.py
  ```
## Extending the System

- **Add new paper sources:**
- **Improve NER/query expansion:**
- **Customize agentic workflows:**
- **Integrate a multimodal vector store:**

## License
MIT

