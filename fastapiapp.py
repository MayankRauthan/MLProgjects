import torch
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import whisper
import os
from pydub import AudioSegment
import io
from fastapi.staticfiles import StaticFiles

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Whisper model for transcription
whisper_model = whisper.load_model("tiny")

# Setup translation pipeline using Hugging Face's transformers library
model_name = "Helsinki-NLP/opus-mt-hi-en"  # Hindi to English translation model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translator = pipeline("translation", model=model, tokenizer=tokenizer)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

def transcribe_and_translate(audio_file_path):
    try:
        # Transcribe audio using Whisper model
        transcription_result = whisper_model.transcribe(audio_file_path)

        # Extract transcription text
        transcription_text = transcription_result['text']

        # Translate from Hindi to English
        translation_result = translator(transcription_text, max_length=400)

        # Extract and return translated text
        translated_text = translation_result[0]['translation_text']
        
        return translated_text

    except Exception as e:
        logging.error(f"An error occurred during transcription and translation: in{e}")
        return None

@app.post("/transcribe-translate/")
async def transcribe_translate(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        audio_content = await file.read()
        audio_file_path = f"{file.filename}"
        
        # Save the audio content to a temporary file
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(audio_content)

        # Perform transcription and translation
        translation = transcribe_and_translate(audio_file_path)

        # Delete the temporary audio file
        os.remove(audio_file_path)

        if translation:
            print(translation)
            return JSONResponse(content={"translation": translation}, status_code=200)
        else:
            raise HTTPException(status_code=500, detail="Transcription and translation failed.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during transcription and translation.{e} ")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
