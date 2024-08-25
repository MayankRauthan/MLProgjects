import torch
import logging
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import whisper
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Whisper model for transcription
whisper_model = whisper.load_model("tiny")

# Setup translation pipeline using Hugging Face's transformers library
model_name = "Helsinki-NLP/opus-mt-hi-en"  # Hindi to English translation model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translator = pipeline("translation", model=model, tokenizer=tokenizer)

def transcribe_and_translate(audio_file):
    try:
        # Transcribe audio using Whisper model
        transcription_result = whisper_model.transcribe(audio_file)

        # Extract transcription text
        transcription_text = transcription_result['text']

        # Translate from Hindi to English
        translation_result = translator(transcription_text, max_length=400)

        # Extract and return translated text
        translated_text = translation_result[0]['translation_text']
        
        return translated_text

    except Exception as e:
        logging.error(f"An error occurred during transcription and translation: {e}")
        return "Transcription and translation failed."

# Define Gradio interface function
def gradio_interface(audio_file):
    try:
        # Perform transcription and translation
        translation = transcribe_and_translate(audio_file.name)  # audio_file is a File object from Gradio

        # Return translation result
        if translation:
            return translation
        else:
            return "Transcription and translation failed."

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return "An error occurred during transcription and translation."

# Gradio interface setup
inputs = gr.File(label="Upload Audio File")
output = gr.Textbox(label="Translation Result")

# Launch Gradio interface
iface = gr.Interface(fn=gradio_interface, inputs=inputs, outputs=output, title="Whisper Transcription & Translation", description="Upload a Hindi audio file for transcription and translation to English using the Whisper model and Hugging Face's translation pipeline.")
iface.launch()
