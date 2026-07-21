# DataAnalyzerAi

<img src="https://res.cloudinary.com/b0tb1mho/image/upload/v1784651004/wmrc9tkchwggidppyu66.webp"/>

> A RAG-powered AI application for interactive Exploratory Data Analysis (EDA), intelligent dataset querying, and automated visualization.

## Overview

DataAnalyzerAi transforms spreadsheet analysis into an AI-powered conversational experience. Upload an Excel or CSV dataset, and the application automatically processes the data, creates vector embeddings, stores them in a vector database, and uses Retrieval-Augmented Generation (RAG) with a Large Language Model (LLM) to answer questions grounded in your data.

Beyond conversational analytics, the application provides automated Exploratory Data Analysis (EDA), downloadable profiling reports, interactive spreadsheet viewing, and pair plot visualizations to help users understand their datasets without writing code.

---

## Features

- Upload and analyze **CSV** and **Excel (XLS/XLSX)** datasets.
- Interactive spreadsheet viewer for exploring uploaded data.
- Automatic dataset preprocessing and chunking.
- Vector embedding generation for efficient semantic search.
- FAISS-powered Vector Database for fast retrieval.
- Retrieval-Augmented Generation (RAG) pipeline for grounded AI responses.
- LLM-powered chatbot that answers questions using only the uploaded dataset.
- AI suggestions for suitable Machine Learning algorithms based on dataset characteristics.
- Automated Exploratory Data Analysis (EDA).
- Interactive pair plot generation for numerical feature relationships.
- Comprehensive data profiling report generation.
- Downloadable profiling report.
- Natural language querying without writing SQL or Python.
- Fast and intuitive Streamlit-based interface.

---

## How It Works

```text
Upload Dataset
       │
       ▼
Parse & Clean Data
       │
       ▼
Chunk Dataset
       │
       ▼
Generate Embeddings
       │
       ▼
Store in FAISS Vector Database
       │
       ▼
User Question
       │
       ▼
Retrieve Relevant Chunks
       │
       ▼
LLM + RAG
       │
       ▼
Grounded AI Answer
       │
       ├────────► Pair Plot Generation
       │
       └────────► Data Profiling Report
```

---

## Tech Stack

### Frontend

- Streamlit

### Backend

- Python

### AI & RAG

- Large Language Models (LLM)
- Retrieval-Augmented Generation (RAG)
- FAISS Vector Database
- Embeddings

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Reporting

- Data Profiling

---

## Project Walkthrough

### Interactive Data Analysis Workflow

<img src="https://res.cloudinary.com/b0tb1mho/image/upload/v1784650976/lqvbok1btejwilemhmth.webp"/>

---

### Data Driven Pair Plot Generation and EDA Insights

<img src="https://res.cloudinary.com/b0tb1mho/image/upload/v1784650994/mt9nav1mfmkulgqcmeni.webp"/>

---

### Excel Viewer for Uploaded Spreadsheets

<img src="https://res.cloudinary.com/b0tb1mho/image/upload/v1784650991/uhe237le3qkz3mw3j4mo.webp"/>

---

### Downloadable Data Profiling Report

<img src="https://res.cloudinary.com/b0tb1mho/image/upload/v1784651167/sv7cdf15jd3ztthc48ym.webp"/>

---

## Installation

```bash
git clone https://github.com/thebitanpaul/DataAnalyzerAi.git

cd DataAnalyzerAi

pip install -r requirements.txt

streamlit run app.py
```

---

## Usage

1. Launch the Streamlit application.
2. Upload a CSV or Excel dataset.
3. Explore the dataset using the built-in spreadsheet viewer.
4. Ask questions in natural language.
5. Let the RAG pipeline retrieve relevant dataset chunks.
6. Receive grounded responses from the LLM.
7. Generate pair plots for visual exploration.
8. Download the generated data profiling report.

---

## Repository Structure

```text
DataAnalyzerAi/
├── app.py
├── requirements.txt
├── assets/
├── data/
├── reports/
├── utils/
├── vector_store/
└── README.md
```

---

## Future Improvements

- Support multiple datasets in a single session.
- Additional statistical visualizations.
- SQL generation over uploaded datasets.
- Time-series analysis support.
- Automated feature engineering suggestions.
- Interactive dashboard generation.
- Multi-user dataset workspace.
- Cloud vector database integration.

---

## Connect With The Engineer

[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/thebitanpaul)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/thebitanpaul)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/thebitanpaul_)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white)](https://www.facebook.com/thebitanpaul)

---

<div align="center">

**2024 · © phiUture · All Rights Reserved**

</div>
