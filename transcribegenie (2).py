from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import time
import logging
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
from io import BytesIO
from profanityfilter import ProfanityFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SpeechTranscriber:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.pipe = None
        self.profanity_filter = ProfanityFilter()

    def load_model(self, model_id):
        try:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id)
            logging.info(f"Model {model_id} loaded successfully on {self.device}.")
        except Exception as e:
            logging.error(f"Error loading model {model_id}: {e}")
            raise

    def create_asr_pipeline(self):
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=0 if torch.cuda.is_available() else -1,
            )
            logging.info("ASR pipeline created successfully.")
        except Exception as e:
            logging.error(f"Error creating ASR pipeline: {e}")
            raise

    def transcribe_audio(self, audio_data, model_choice, language=None, use_profanity_filter=False):
        model_id = f"openai/whisper-{model_choice}"
        try:
            self.load_model(model_id)
            self.create_asr_pipeline()

            audio = AudioSegment.from_file(BytesIO(audio_data))
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio_array = np.array(audio.get_array_of_samples())

            audio_array = audio_array.astype(np.float32) / np.max(np.abs(audio_array))

            try:
                start_time = time.time()
                progress = 0
                while progress < 100:
                    time.sleep(1)
                    progress += 10
                result = self.pipe(audio_array, generate_kwargs={"language": language} if language else {})
                end_time = time.time()
                runtime = end_time - start_time
            except Exception as e:
                logging.error(f"Error during transcription: {e}")
                return f"Error during transcription: {e}", None, None, None

            transcription_text = result.get("text", "")
            profanity_detected = False

            if use_profanity_filter:
                profanity_detected = self.profanity_filter.is_profane(transcription_text)
                if profanity_detected:
                    transcription_text = ' '.join(['*' * len(word) if self.profanity_filter.is_profane(word) else word for word in transcription_text.split()])

            word_count = len(transcription_text.split())

            logging.info(f"Transcription: {transcription_text}")
            logging.info(f"Profanity detected: {profanity_detected}")
            logging.info(f"Runtime: {runtime} seconds")
            logging.info(f"Word Count: {word_count}")

            return transcription_text, runtime, word_count, profanity_detected

        except Exception as e:
            logging.error(f"Error in transcribe_audio: {e}")
            return f"Error in transcribe_audio: {e}", None, None, None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speech Transcription</title>
    <style>
        body {
            background-color: #1e1e2e;
            color: #ffffff;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            overflow: auto;
        }
        .container {
            width: 90%;
            max-width: 900px;
            padding: 20px;
            background-color: #1e1e2e;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
            position: relative;
            margin: 20px auto;
            box-sizing: border-box;
        }
        h1 {
            font-size: 32px;
            margin: 0;
            color: #ffffff;
            text-align: center;
            border-bottom: 2px solid #007bff;
            padding: 20px 0;
            background-color: #1e1e2e;
            border-radius: 10px 10px 0 0;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .upload-container, .result-container {
            margin-bottom: 20px;
            border: 1px solid #ffffff;
            border-radius: 5px;
            padding: 10px;
            background-color: #1e1e2e;
        }
        .upload-label, .result-label {
            display: block;
            margin-bottom: 8px;
            font-size: 18px;
            color: #ffffff;
        }
        input[type="file"], input[type="text"], select, button {
            width: calc(100% - 24px);
            padding: 12px;
            border: 1px solid #ffffff;
            border-radius: 5px;
            margin-bottom: 12px;
            box-sizing: border-box;
            background-color: #1e1e2e;
            color: #ffffff;
        }
        button {
            padding: 12px;
            border: 1px solid #ffffff;
            border-radius: 5px;
            background-color: #007bff;
            color: #ffffff;
            cursor: pointer;
            font-size: 16px;
            text-transform: uppercase;
        }
        button#clearButton {
            background-color: #6c757d;
        }
        .result-container {
            max-height: 300px;
            overflow-y: auto;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #ffffff;
            border-radius: 5px;
            background-color: #1e1e2e;
            color: #ffffff;
            margin-bottom: 12px;
            box-sizing: border-box;
            resize: vertical;
            overflow: hidden;
        }
        #progressBarContainer {
            display: none;
            width: 100%;
            background-color: #1e1e2e;
            padding: 10px;
            box-sizing: border-box;
            text-align: center;
        }
        #progressBar {
            width: 80%;
            max-width: 600px;
            background: #444;
            border-radius: 5px;
            overflow: hidden;
            height: 16px;
            border: 1px solid #ffffff;
            margin: 0 auto;
            position: relative;
        }
        #progressBarFill {
            height: 100%;
            width: 0;
            background-color: #007bff;
            text-align: center;
            line-height: 16px;
            color: #ffffff;
            font-size: 12px;
            transition: width 0.5s ease;
        }
        #downloadButton {
            display: none;
            padding: 12px;
            background-color: #007bff;
            color: #ffffff;
            border-radius: 5px;
            text-align: center;
            text-decoration: none;
            margin-top: 12px;
            font-size: 16px;
            border: 1px solid #ffffff;
        }
        .filter-container {
            margin-bottom: 12px;
        }
        #profanityFilter {
            width: auto;
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Speech Transcription</h1>
        <div class="upload-container">
            <label for="audioInput" class="upload-label">Upload Audio</label>
            <input type="file" id="audioInput" accept="audio/*">
            <label for="modelChoice" class="upload-label">Model Choice</label>
            <select id="modelChoice">
                <option value="tiny">tiny</option>
                <option value="base">base</option>
                <option value="small">small</option>
                <option value="medium">medium</option>
                <option value="large-v3">large-v3</option>
            </select>
            <label for="languageInput" class="upload-label">Language (e.g., English, Hindi, Marathi, Tamil)</label>
            <input type="text" id="languageInput" placeholder="Enter the language for transcription">
        </div>
        <div class="filter-container">
            <input type="checkbox" id="profanityFilter" name="profanityFilter">
            <label for="profanityFilter">Enable Profanity Filter</label>
        </div>
        <div class="buttons">
            <button id="clearButton">Clear</button>
            <button id="submitButton">Transcribe</button>
        </div>
        <div id="progressBarContainer">
            <label class="result-label">Processing...</label>
            <div id="progressBar">
                <div id="progressBarFill">0%</div>
            </div>
        </div>
        <div class="result-containers">
            <label for="transcriptionResult" class="result-label">Transcription</label>
            <textarea id="transcriptionResult" rows="10" readonly></textarea>
            <label for="runtimeResult" class="result-label">Process Time (seconds)</label>
            <input type="text" id="runtimeResult" readonly>
            <label for="wordCountResult" class="result-label">Word Count</label>
            <input type="text" id="wordCountResult" readonly>
            <label for="profanityDetectedResult" class="result-label">Profanity Detected</label>
            <input type="text" id="profanityDetectedResult" readonly>
            <a id="downloadButton" href="#" download="transcription.txt">Download Transcription</a>
        </div>
    </div>
    <script>
        document.getElementById('submitButton').addEventListener('click', async () => {
            const audioInput = document.getElementById('audioInput').files[0];
            const modelChoice = document.getElementById('modelChoice').value;
            const language = document.getElementById('languageInput').value;
            const profanityFilter = document.getElementById('profanityFilter').checked;

            if (!audioInput) {
                alert('Please upload an audio file.');
                return;
            }

            const formData = new FormData();
            formData.append('audio_file', audioInput);
            formData.append('model_choice', modelChoice);
            formData.append('language', language);
            formData.append('profanity_filter', profanityFilter);

            const progressBarContainer = document.getElementById('progressBarContainer');
            const progressBarFill = document.getElementById('progressBarFill');

            progressBarContainer.style.display = 'block';
            progressBarFill.style.width = '0%';
            progressBarFill.textContent = '0%';

            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const data = await response.json();
                document.getElementById('transcriptionResult').textContent = data.transcription;
                document.getElementById('runtimeResult').value = data.runtime;
                document.getElementById('wordCountResult').value = data.word_count;
                document.getElementById('profanityDetectedResult').value = data.profanity_detected;

                const downloadButton = document.getElementById('downloadButton');
                downloadButton.href = 'data:text/plain;base64,' + btoa(data.transcription);
                downloadButton.style.display = 'block';

                progressBarFill.style.width = '100%';
                progressBarFill.textContent = '100%';
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during the transcription process.');
            } finally {
                setTimeout(() => {
                    progressBarContainer.style.display = 'none';
                }, 500);
            }
        });

        document.getElementById('clearButton').addEventListener('click', () => {
            document.getElementById('audioInput').value = '';
            document.getElementById('languageInput').value = '';
            document.getElementById('transcriptionResult').textContent = '';
            document.getElementById('runtimeResult').value = '';
            document.getElementById('wordCountResult').value = '';
            document.getElementById('profanityDetectedResult').value = '';
            document.getElementById('downloadButton').style.display = 'none';
            document.getElementById('progressBarContainer').style.display = 'none';
            document.getElementById('progressBarFill').style.width = '0%';
            document.getElementById('progressBarFill').textContent = '0%';
            document.getElementById('profanityFilter').checked = false;
        });
    </script>
</body>
</html>
    """

class TranscriptionRequest(BaseModel):
    model_choice: str
    language: Optional[str] = None
    profanity_filter: bool = False

@app.post("/transcribe", response_model=dict)
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model_choice: str = Form(...),
    language: Optional[str] = Form(None),
    profanity_filter: bool = Form(False)
):
    transcriber = SpeechTranscriber()
    audio_data = await audio_file.read()

    transcription, runtime, word_count, profanity_detected = transcriber.transcribe_audio(audio_data, model_choice, language, profanity_filter)
    return JSONResponse({
        "transcription": transcription,
        "runtime": runtime,
        "word_count": word_count,
        "profanity_detected": profanity_detected
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)