import streamlit as st
import yt_dlp
import requests
import webvtt
import re
from io import BytesIO
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("⚡ Instant YouTube Transcript Extractor")

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

# UI
url = st.text_input("🔗 YouTube Video URL", placeholder="https://www.youtube.com/watch?v=VIDEO_ID")

if st.button("🚀 Get Transcript"):
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
    else:
        st.warning("Please enter a valid YouTube URL.")
