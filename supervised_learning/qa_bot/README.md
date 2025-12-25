# QA Bot — Semantic Question Answering System

This project implements a **semantic question answering bot** that can extract answers from a corpus of reference documents using a combination of **transformer-based extractive QA** and **semantic search**.

It includes:
- Single-document question answering
- Interactive QA loop
- Semantic similarity document search
- Multi-document QA chatbot

---

## 📁 Project Structure

qa_bot/
├── 0-qa.py # Task 0 — basic QA function
├── 0-main.py # Test script for Task 0
├── 1-loop.py # Task 1 — interactive loop
├── 1-main.py # Test script for Task 1
├── 2-qa.py # Task 2 — QA with fallback answers
├── 2-main.py # Test script for Task 2
├── 3-semantic_search.py # Task 3 — semantic document search
├── 3-main.py # Test script for Task 3
├── 4-qa.py # Task 4 — QA across full corpus
├── 4-main.py # Test script for Task 4
├── ZendeskArticles/ # Reference documents (corpus)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── venv/ # Virtual environment (not committed)

install Dependencies
 
pip install --upgrade pip
pip install tensorflow==2.15.0 tensorflow-hub==0.15.0
pip install transformers==4.44.2 sentence-transformers
