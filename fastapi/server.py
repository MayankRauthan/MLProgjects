import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Whisper model for transcription
whisper_model = whisper.load_model("large")

# Setup translation pipeline using Hugging Face's transformers library
model_name = "Helsinki-NLP/opus-mt-hi-en"  # Hindi to English translation model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translator = pipeline("translation", model=model, tokenizer=tokenizer)

app = FastAPI()

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
        logging.error(f"An error occurred during transcription and translation: {e}")
        return "Transcription and translation failed."

@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Whisper Transcription & Translation</title>
        <style>
        body {
    background-color: #1e1e2e;
    color: #ffffff;
    font-family: Arial, sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
}

.container {
    text-align: center;
    max-width: 600px;
    width: 100%;
    padding: 20px;
    background-color: #28293e;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

h1 {
    font-size: 24px;
    margin-bottom: 16px;
}

p {
    margin-bottom: 24px;
}

.upload-container {
    margin-bottom: 24px;
}

.upload-label {
    display: block;
    margin-bottom: 8px;
    font-size: 18px;
}

#audioFile {
    width: 100%;
    padding: 12px;
    border: 1px solid #ffffff;
    border-radius: 4px;
    background-color: #1e1e2e;
    color: #ffffff;
}

.buttons {
    display: flex;
    justify-content: space-between;
    margin-bottom: 24px;
}

button {
    padding: 12px 24px;
    border: none;
    border-radius: 4px;
    background-color: #ff6f61;
    color: #ffffff;
    cursor: pointer;
}

button#clearButton {
    background-color: #444466;
}

.result-container {
    text-align: left;
}

.result-label {
    display: block;
    margin-bottom: 8px;
    font-size: 18px;
}

#result {
    width: 100%;
    height: 100px;
    padding: 12px;
    border: 1px solid #ffffff;
    border-radius: 4px;
    background-color: #1e1e2e;
    color: #ffffff;
    resize: none;
}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Whisper Transcription & Translation</h1>
            <p>Upload a Hindi audio file for transcription and translation to English</p>
            
            <div class="upload-container">
                <label for="audioFile" class="upload-label">Upload Audio File</label>
                <input type="file" id="audioFile" accept="audio/*">
            </div>
            
            <div class="buttons">
                <button id="clearButton">Clear</button>
                <button id="submitButton">Submit</button>
            </div>
            
            <div class="result-container">
                <label for="result" class="result-label">Translation Result</label>
                <textarea id="result" readonly></textarea>
            </div>
        </div>
        <script>
        document.getElementById('submitButton').addEventListener('click', async () => {
    const audioFileInput = document.getElementById('audioFile');
    const file = audioFileInput.files[0];
    
    if (file) {
        document.getElementById('result').value = "Proessing...";

        const formData = new FormData();
        formData.append('audio_file', file);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const result = await response.text();
        document.getElementById('result').value = result;
    } else {
        alert('Please select an audio file.');
    }
});

document.getElementById('clearButton').addEventListener('click', () => {
    document.getElementById('audioFile').value = '';
    document.getElementById('result').value = '';
});
        </script>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_file(audio_file: UploadFile = File(...)):
    try:
        audio_file_path = f"{audio_file.filename}"
        with open(audio_file_path, "wb") as f:
            f.write(await audio_file.read())
        
        translation = transcribe_and_translate(audio_file_path)
        return translation

    except Exception as e:
        logging.error(f"An error occurred during file upload and processing: {e}")
        return "An error occurred during transcription and translation."

# Serve static files
@app.get("/static/{filename}", response_class=HTMLResponse)
async def static_files(filename: str):
    static_folder = os.path.join(os.path.dirname(__file__), "static")
    file_path = os.path.join(static_folder, filename)
    with open(file_path) as file:
        return HTMLResponse(file.read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
