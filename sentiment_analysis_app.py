import streamlit as st
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load a pre-trained sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    try:
        # Perform sentiment analysis
        result = classifier(text)
        # Format and return the result
        return result
    except Exception as e:
        logger.error(f"An error occurred during sentiment analysis: {str(e)}")
        return None

# Streamlit UI
st.title("Sentiment Analysis App")

# File upload
uploaded_file = st.file_uploader("Upload a text file for sentiment analysis:", type="txt")

if uploaded_file is not None:
    try:
        # Read text from uploaded file
        file_text = uploaded_file.read().decode("utf-8")
        result = analyze_sentiment(file_text)
        if result:
            st.write("Sentiment Analysis Result:")
            for r in result:
                st.write(f"Label: {r['label']}, Score: {round(r['score'], 4)}")
        else:
            st.error("An error occurred during sentiment analysis.")
    except Exception as e:
        logger.error(f"An error occurred while reading the file: {str(e)}")
        st.error("An error occurred while reading the file.")
