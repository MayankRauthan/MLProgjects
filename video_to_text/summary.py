from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
origins = [
    "http://127.0.0.1:5500",  # Add your frontend origin here
    "http://localhost:5500",  # Add your frontend origin here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Extract Audio from Video
def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

# Transcribe Audio to Text with Chunking
def transcribe_chunk(chunk):
    recognizer = sr.Recognizer()
    with sr.AudioFile(chunk) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            text = ""
    return text

def transcribe_audio(audio_path, chunk_length_ms=60000):
    audio = AudioSegment.from_wav(audio_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_files.append(chunk_path)

    with ThreadPoolExecutor() as executor:
        transcriptions = list(executor.map(transcribe_chunk, chunk_files))

    # Cleanup temp chunk files
    for chunk_file in chunk_files:
        os.remove(chunk_file)

    return " ".join(transcriptions)

# Summarize the Text with Adjustable Length
def summarize_text(text, max_length, min_length):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Summarize Long Text in Chunks
def summarize_long_text(text, length_ratio=1):
    words = text.split()
    chunk_len = len(words)

    max_length = int(chunk_len * length_ratio)
    min_length = int(max_length * 0.7)

    summary = summarize_text(text, max_length=max_length, min_length=min_length)

    return summary

# Generate Video Summary
def generate_video_summary(video_path):
    # Paths
    audio_path = "temp_audio.wav"

    # Extract audio from video
    extract_audio(video_path, audio_path)

    # Transcribe audio to text
    text = transcribe_audio(audio_path)

    # Summarize the text
    summary = summarize_long_text(text)

    # Cleanup temp audio file
    os.remove(audio_path)

    return summary

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    print("function called")
    video_path = f"temp_video_{file.filename}"
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    summary = generate_video_summary(video_path)
    print(summary)
    # Cleanup temp video file
    os.remove(video_path)

    return {"summary": summary}

@app.get("/")
async def main():
    content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video Summary</title>
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

        #videoFile {
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
            <h1>Video Summary</h1>
            <p>Upload a video file for summary generation</p>
            
            <div class="upload-container">
                <label for="videoFile" class="upload-label">Upload Video File</label>
                <input type="file" id="videoFile" accept="video/*">
            </div>
            
            <div class="buttons">
                <button id="clearButton">Clear</button>
                <button id="submitButton">Submit</button>
            </div>
            
            <div class="result-container">
                <label for="result" class="result-label">Summary Result</label>
                <textarea id="result" readonly></textarea>
            </div>
        </div>
        <script>
        document.getElementById('submitButton').addEventListener('click', async () => {
            const videoFileInput = document.getElementById('videoFile');
            const file = videoFileInput.files[0];
            
            if (file) {
                document.getElementById('result').value = "Processing...";

                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/upload-video/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                document.getElementById('result').value = result.summary;
            } else {
                alert('Please select a video file.');
            }
        });

        document.getElementById('clearButton').addEventListener('click', () => {
            document.getElementById('videoFile').value = '';
            document.getElementById('result').value = '';
        });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
