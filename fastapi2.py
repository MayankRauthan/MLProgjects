from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import whisper
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Whisper model for transcription
whisper_model = whisper.load_model("tiny")  # Or choose a different model size

# Setup translation pipeline using Hugging Face's transformers library
model_name = "Helsinki-NLP/opus-mt-hi-en"  # Hindi to English translation model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translator = pipeline("translation", model=model, tokenizer=tokenizer)

app = FastAPI()

@app.post("/transcribe_and_translate")
async def transcribe_translate(audio_file: UploadFile = File(...)):
    try:
        # Validate file type (optional, but recommended for security)
        if audio_file.content_type not in ["audio/mpeg", "audio/wav"]:
            raise HTTPException(status_code=400, detail="Invalid audio file format. Supported formats: MP3, WAV")

        # Save the uploaded audio file temporarily
        file_path = f"uploads/{audio_file.filename}"  # Adjust path if needed
        with open(file_path, "wb") as buffer:
            buffer.write(await audio_file.read())

        # Perform transcription using Whisper model
        transcription_result = whisper_model.transcribe(file_path)

        # Extract transcription text
        transcription_text = transcription_result["text"]

        # Translate from Hindi to English
        translation_result = translator(transcription_text, max_length=400)

        # Extract and return translated text
        translated_text = translation_result[0]["translation_text"]

        # Clean up the temporary audio file
        os.remove(file_path)

        return {"translated_text": translated_text}

    except Exception as e:
        logging.error(f"An error occurred during transcription and translation: {e}")
        return {"error": "Transcription and translation failed."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Adjust host and port as needed
