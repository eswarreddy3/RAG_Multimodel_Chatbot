# 🤖 RAG ChatBot

RAG ChatBot is a Streamlit-based application that combines Retrieval Augmented Generation (RAG) with various data sources like PDFs, YouTube videos, and resumes. It leverages OpenAI's GPT models and LangChain for intelligent, context-aware interactions.

## ✨ Key Features

- **Multiple Data Sources:** PDF documents, YouTube videos, and resume analysis
- **RAG Pipeline:** Uses embeddings and Chroma for context-based answers
- **Resume Evaluation:** Reviews structure, missing sections, and provides improvement suggestions
- **YouTube Transcript Extraction:** Extracts and displays subtitles from videos
- **Interactive UI:** Built with Streamlit for easy use
- **Chunked PDF Parsing:** For efficient vector search

## 🚀 Installation

### Prerequisites

- Python 3.8+
- OpenAI API Key

### Setup Instructions

```bash
git clone https://github.com/your_username/rag_chatbot.git
cd rag_chatbot
pip install -r requirements.txt
```

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Run the app:

```bash
streamlit run main.py
```

## 📋 Dependencies

See `requirements.txt`, which includes:

- `langchain`, `langchain_openai`, `openai`
- `streamlit`, `chromadb`, `faiss-cpu`
- `pypdf`, `pdfplumber`, `webvtt-py`
- `yt-dlp`, `unstructured`, `python-dotenv`, `tqdm`

## 💻 Application Modes

1. **Ask with LLM** – Directly ask GPT without any document.
2. **Resume Upload** – Analyze resumes and receive structured feedback.
3. **YouTube URL** – Extract transcripts and download them.
4. **PDF Upload** – Upload PDFs and ask questions using RAG.

## 🏗️ Architecture Overview

- **LLM:** `ChatOpenAI` (gpt-3.5-turbo)
- **Embeddings:** `OpenAIEmbeddings`
- **Vector Store:** `Chroma`
- **Chunking:** `RecursiveCharacterTextSplitter`
- **PDF Loading:** `PyPDFLoader`
- **YouTube Transcripts:** `yt-dlp + webvtt-py`

## 📁 Project Structure

```
project/
├── main.py              # Main Streamlit app
├── requirements.txt     # Python dependencies
├── .env                 # Environment config
└── README.md            # Project documentation
```

## 🎯 Use Cases

- Resume review for job applications
- Extracting notes from YouTube lectures
- QA over research papers and documentation

## ⚠️ Limitations

- YouTube videos must have English subtitles
- Best suited for text-based PDFs
- Requires an OpenAI API key
- Network connection is mandatory

## 🔍 Troubleshooting

- Install missing packages: `pip install -r requirements.txt --upgrade`
- OpenAI errors: Check `.env` and API usage limits
- YouTube failures: Ensure video has subtitles

## 🔐 Security

- Store API keys in `.env`, not in code
- Validate all file uploads
- Monitor API usage for cost management

---

MIT License © 2025 Your Name