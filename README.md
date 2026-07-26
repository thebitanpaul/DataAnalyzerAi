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

## About phiUture


Welcome to phiUture — Beautiful Technology. Centered Around You. Shaping Tomorrow.

Inspired by the Golden Ratio (φ), our name represents beautiful, intelligent engineering, while the "U" stands for You—placing people at the center of every solution we build.

phiUture is an AI-first software company creating intelligent products, automation systems, and modern digital experiences. This channel documents the journey of building practical AI solutions, from concept to deployment.

```text
Here you'll find:
• AI applications and product demos
• AI agents and automation workflows
• Web and mobile app showcases
• Machine Learning and Data Engineering projects
• Product launches and development insights
• UI/UX and software engineering content
• Tutorials, experiments, and future innovations
```

| Personal | Business | Artist |
|----------|----------|--------|
| [![GitHub](https://img.shields.io/badge/GitHub-thebitanpaul-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/thebitanpaul) | [![Website](https://img.shields.io/badge/Website-phiUture-000000?style=for-the-badge&logo=googlechrome&logoColor=white)](https://phiuture.com) | [![YouTube](https://img.shields.io/badge/YouTube-thebitanpaul-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@thebitanpaul) |
| [![LinkedIn](https://img.shields.io/badge/LinkedIn-thebitanpaul-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/thebitanpaul) | [![YouTube](https://img.shields.io/badge/YouTube-phiUture-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@phiuture) | [![Spotify](https://img.shields.io/badge/Spotify-1DB954?style=for-the-badge&logo=spotify&logoColor=white)](https://open.spotify.com/artist/6ghDcCBlKzJIgm3e586jpV) |
| [![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/thebitanpaul) | [![Google Play](https://img.shields.io/badge/Google_Play-Developer-34A853?style=for-the-badge&logo=googleplay&logoColor=white)](https://play.google.com/store/apps/dev?id=6358474525178045834&hl=en) | [![YouTube Music](https://img.shields.io/badge/YouTube_Music-FF0000?style=for-the-badge&logo=youtubemusic&logoColor=white)](https://music.youtube.com/playlist?list=OLAK5uy_km3cjEB2zl59Etcgv9UBKWw800O9G3NdE) |
| [![Facebook](https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white)](https://facebook.com/thebitanpaul) | [![Email](https://img.shields.io/badge/Business_Email-thephiuture%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:thephiuture@gmail.com) | [![Amazon Music](https://img.shields.io/badge/Amazon_Music-46C3D0?style=for-the-badge&logo=amazonmusic&logoColor=white)](https://music.amazon.com/albums/B0G52QMYDC) |
| [![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/thebitanpaul_) |  | [![Apple Music](https://img.shields.io/badge/Apple_Music-FA243C?style=for-the-badge&logo=applemusic&logoColor=white)](https://music.apple.com/us/artist/thebitanpaul/1858534880) |
| [![Snapchat](https://img.shields.io/badge/Snapchat-FFFC00?style=for-the-badge&logo=snapchat&logoColor=000000)](https://snapchat.com/t/UgO0Iywr) |  | [![JioSaavn](https://img.shields.io/badge/JioSaavn-2BC5B4?style=for-the-badge&logo=jiosaavn&logoColor=white)](https://www.jiosaavn.com/artist/thebitanpaul-songs/zuo0NgC65gQ_) |
| [![Email](https://img.shields.io/badge/Personal_Email-thebitanpaul%40gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:thebitanpaul@gmail.com) |  |  |


---

<div align="center">

**2024 · © phiUture · All Rights Reserved**

</div>
