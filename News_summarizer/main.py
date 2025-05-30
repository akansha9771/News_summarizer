import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load .env variables
load_dotenv()

# Set up Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Prompt template
summarize_prompt = PromptTemplate(
    template="Summarize the following news article:\n\n{article}\n\nSummary:",
    input_variables=["article"]
)

# LLM chain
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# News extractor
def extract_news(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"❌ Failed to fetch news from {url}: {e}"

# Summarize article
def summarize_news(url):
    article = extract_news(url)
    if article.startswith("❌"):
        return article
    summary = summarize_chain.run(article=article)
    return summary

# --- Streamlit UI ---
st.set_page_config(page_title="📰 News Summarizer", page_icon="🧠", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #fdf6f0;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0,0,0,0.05);
    }
    .stTextInput > div > div > input {
        background-color: #fff2e6;
        border: 1px solid #ffa94d;
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #ffa94d;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stButton > button:hover {
        background-color: #ff922b;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# UI Layout
with st.container():
    st.markdown("## 🌟 Welcome to the AI News Summarizer")
    st.markdown("Effortlessly summarize any news article using Google's Gemini AI!")

    st.markdown("### 🔗 Paste a News URL Below")
    url = st.text_input("Enter the news article URL")

    if st.button("✨ Summarize"):
        if url:
            with st.spinner("⏳ Summarizing... please wait..."):
                result = summarize_news(url)
            st.success("✅ Summary Ready!")
            st.markdown("### 📝 Summary:")
            st.markdown(f"<div style='background-color:#fff3cd;padding:1em;border-radius:10px'>{result}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a valid URL.")
