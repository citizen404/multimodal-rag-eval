Multimodal RAG & Evaluation Prototype
Overview

This project implements a lightweight, evaluation-driven multimodal RAG system for question answering over complex PDF documents containing text, tables, and images.

The primary goal is architectural clarity, controllability, and reproducibility, rather than maximizing raw model performance.
The system is designed as a clean reference implementation of a production-oriented RAG pipeline with explicit ingestion, retrieval, generation, and evaluation stages.

"""
Architecture

End-to-end pipeline:
PDF → Markdown → Chunking → Embeddings → Vector Store → Retrieval → Controlled Generation → Evaluation

Key design principles

Text-only retrieval layer for simplicity, debuggability, and reproducible evaluation
Explicit separation between ingestion, retrieval, generation, and evaluation
Retrieval-first architecture with no hidden or duplicated context fetching
Evaluation as a first-class component, not an afterthought

"""
Components

1. Ingestion Pipeline

The ingestion module converts complex PDF documents into a unified, text-based knowledge representation.
Text and tables are extracted into Markdown using pymupdf4llm, preserving document structure.
Images and charts are converted into structured textual descriptions using LLM-based captioning.
This enables reuse of a standard text-only RAG pipeline while still capturing non-textual information.

Chunking strategy:
Structural splitting by Markdown headers
Recursive character-based splitting for size control
Each chunk is enriched with metadata (source document, content type) to support future retrieval strategies and analysis.

Design trade-off:
Multimodal embeddings were intentionally avoided to keep retrieval transparent, controllable, and easier to evaluate.

2. RAG Pipeline

Vector store: Chroma (persistent)
Embeddings: text-embedding-3-small
LLM: gpt-4o-mini

The RAG system follows a controlled retrieval-first pattern:
Documents are retrieved exactly once from the vector store.
Retrieved chunks are explicitly injected into the generation prompt.
If no relevant context is found, the system reports insufficient information instead of hallucinating.
This approach improves:
reproducibility,
debuggability,
evaluation reliability.

All secrets are managed via environment variables, ensuring portability across environments.

3. Evaluation

The project includes an automated evaluation pipeline based on RAGAS.
Curated question set derived from source documents
Ground-truth answers manually defined

Metrics:
Faithfulness
Answer Relevancy
Context Recall

Evaluation results are exported to CSV for reproducibility and comparison across runs.
This setup enables systematic measurement of retrieval quality and answer grounding rather than subjective inspection.

"""

4. CLI Interface

A simple CLI interface is provided for interactive querying.
On first run, documents are parsed and indexed.
Subsequent runs reuse the persisted vector store.
Each answer is returned together with its source references.

"""
Setup

1. #Create  a data/ folder in the root directory and put there all your PDF's

2. #install environment
pip install -r requirements.txt

Configure API Key: Create a .env file in the root directory and add your key:
OPENAI_API_KEY=sk-proj-your_key_here

3. #create a DB:
python3 ingestion.py

4. #run the script:
python3 main.py

"""
Evaluation Framework

python3 evaluate_rag.py

Results are saved to evaluation_report.csv.

"""

Prompting Strategy

The generation step uses a strict system prompt to minimize hallucinations:
- Answers are based only on retrieved context
- No invented facts, numbers, or names
- Explicit handling of unanswerable questions

This prompt is intentionally conservative to support evaluation-driven iteration.