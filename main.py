try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass


import os
import tempfile
import streamlit as st
import yt_dlp
import requests
import webvtt
import re
from io import BytesIO
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


#QueryVerse — One AI to rule all your content: PDFs, videos, and more.

# Load API Key
load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Streamlit App Title
st.title("🤖 RAG ChatBot")

# Model & Embedding Initialization
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                              api_key=OPENAI_API_KEY)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant. Use the following context to answer the user.\nContext:\n{context}"),
    ("user", "{input}")
])

# Sidebar source selection
source = st.sidebar.selectbox("Select Retrieval Source", ["Document Retriver", "Resume Analyzer", "YouTube Video Summarizer", "Ask to LLM"])

# Based on selected source, show input
upload = None
url = None

# 📺 Custom function to get transcript from YouTube
@st.cache_data(show_spinner=False)
def fast_extract_transcript(video_url: str):
    try:
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'quiet': True,
            'no_warnings': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            title = info.get("title", "Unknown Title")
            duration = info.get("duration", 0)
            subs = info.get('subtitles') or info.get('automatic_captions', {})

            for lang in ['en', 'en-US', 'en-GB']:
                if lang in subs:
                    for entry in subs[lang]:
                        if entry['ext'] == 'vtt':
                            url = entry['url']
                            response = requests.get(url, timeout=10)
                            if response.ok:
                                # Parse directly from memory
                                vtt_buffer = BytesIO(response.content)
                                text = ""
                                for caption in webvtt.read_buffer(vtt_buffer):
                                    clean = re.sub(r'<[^>]+>', '', caption.text).strip()
                                    text += clean + " "
                                return {
                                    "title": title,
                                    "duration": duration,
                                    "language": lang,
                                    "content": text.strip()
                                }
        return None
    except Exception as e:
        st.error(f"Error extracting transcript: {str(e)}")
        return None


if source == "YouTube Video Summarizer":
    st.subheader("🎥 Enter YouTube URL")
    url = st.text_input("YouTube URL:")
elif source == "Document Retriver":
    st.subheader("📄 Upload PDF for Retrieval")
    upload = st.file_uploader("Upload a PDF file", type="pdf")
    user_query = st.text_input("💡 Ask a question:")
elif source == "Resume Analyzer":
    st.subheader("📄 Upload Resume for Retrieval")
    upload = st.file_uploader("Upload a Resume file", type="pdf")
    experience = st.selectbox("🧑‍💼 Select your experience", ['Fresher', '0 - 2 years', '3 - 5 years', '5+ years'])
else:
    st.subheader("💬 Ask anything, using only LLM (no RAG)")
    user_query = st.text_input("💡 Ask a question:")

# Query input
submit = st.button("🔍 Submit")

# On submit
if submit:
    # --- PDF Upload Path ---
    if source == "Document Retriver" and upload and user_query:
        with st.spinner("📄 Processing PDF..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(upload.read())
                    temp_path = tmp_file.name

                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)

                # ✅ Fresh Chroma store for each upload
                with tempfile.TemporaryDirectory() as temp_dir:
                    db = Chroma.from_documents(chunks, embeddings, persist_directory=temp_dir)
                    retriever = db.as_retriever()

                    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
                    result = rag_chain.invoke({"input": user_query})
                    
                st.markdown("### ✅ Answer from PDF")
                st.write(result["answer"])

                os.remove(temp_path)
            except Exception as e:
                st.error(f"❌ Error processing PDF: {e}")

    
    # --- Resume Upload ---
    elif source == "Resume Analyzer" and upload:
        with st.spinner("🎥 Processing Resume..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(upload.read())
                    temp_path = tmp_file.name

                # Load resume text
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                resume_text = "\n".join([doc.page_content for doc in docs])

                # Define evaluation prompt
                evaluation_prompt = PromptTemplate.from_template("""
                You are a professional resume reviewer. Below is a resume and the candidate's experience level is: {experience}.

                Provide a concise evaluation in clear bullet points or structured sections:
                1. ✅ Structure Suitability – Is it appropriate for their experience level?
                2. ❌ Missing Sections – Mention only the important ones.
                3. ✍️ Suggestions for Improvement – Focused and actionable.
                4. 📊 Overall Rating – Give a score in percentage (e.g., 82%).

                Resume Content:
                {resume_text}

                Respond only with the formatted review, no extra explanations or summaries.
                """)

                evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)
                result = evaluation_chain.run({
                    "resume_text": resume_text,
                    "experience": experience
                })

                st.markdown("### 📝 Resume Evaluation")
                st.write(result)

                os.remove(temp_path)

            except Exception as e:
                st.error(f"❌ Error processing YouTube URL: {e}")
    
    # --- YouTube URL Path ---
    elif source == "YouTube Video Summarizer" and url:
        if url:
            with st.spinner("Fetching transcript..."):
                result = fast_extract_transcript(url)

            if result:
                st.success("Transcript extracted successfully!")
                st.markdown(f"**🎬 Title:** {result['title']}")

                word_count = len(result['content'].split())
                st.markdown(f"**📝 Word Count:** {word_count}")

                with st.expander("📖 View Transcript", expanded=True):
                    st.text_area("Transcript", result['content'], height=400)

                st.download_button("📥 Download Transcript", result['content'],
                                file_name=f"{result['title']}_transcript.txt", mime="text/plain")
            else:
                st.error("No subtitles found or video is restricted.")


    # --- Default to LLM only ---
    else:
        with st.spinner("🤖 Generating answer..."):
            try:
                result = llm.invoke(user_query)
                st.markdown("### 💬 Answer from LLM")
                st.write(result.content)
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")

else:
    st.warning("Please enter a question before submitting.")
