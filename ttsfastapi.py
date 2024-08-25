from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load models
preload_models()

class CustomTextToSpeechPipeline:
    def __init__(self):
        # Load models
        preload_models()

    def __call__(self, text):
        # Generate audio from text
        audio_array = generate_audio(text)
        return {
            "audio": audio_array,
            "sampling_rate": SAMPLE_RATE
        }

# Initialize the custom pipeline
synthesiser = CustomTextToSpeechPipeline()

@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bark Text-to-Speech</title>
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

        #textInput {
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

        #audioResult {
            width: 100%;
        }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Bark Text-to-Speech</h1>
            <p>Enter text to generate speech using the Bark model</p>
            
            <div class="upload-container">
                <label for="textInput" class="upload-label">Enter Text</label>
                <textarea id="textInput" rows="5" placeholder="Enter text here..."></textarea>
            </div>
            
            <div class="buttons">
                <button id="clearButton">Clear</button>
                <button id="submitButton">Submit</button>
            </div>
            
            <div class="result-container">
                <label for="audioResult" class="result-label">Audio Result</label>
                <audio id="audioResult" controls></audio>
            </div>
        </div>
        <script>
        document.getElementById('submitButton').addEventListener('click', async () => {
            const text = document.getElementById('textInput').value;
            if (text) {
                document.getElementById('audioResult').src = '';
                const response = await fetch('/synthesize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: text })
                });
                const audioFilePath = await response.text();
                document.getElementById('audioResult').src = audioFilePath;
            } else {
                alert('Please enter some text.');
            }
        });

        document.getElementById('clearButton').addEventListener('click', () => {
            document.getElementById('textInput').value = '';
            document.getElementById('audioResult').src = '';
        });
        </script>
    </body>
    </html>
    """

@app.post("/synthesize")
async def synthesize_text(request: dict):
    try:
        text = request["text"]
        speech = synthesiser(text)

        # Extract the audio array and sampling rate from the output
        audio = speech["audio"]
        sampling_rate = speech["sampling_rate"]

        # Save the audio to a WAV file
        file_path = "bark_generation.wav"
        write_wav(file_path, sampling_rate, audio)

        return file_path

    except Exception as e:
        return f"An error occurred during synthesis: {e}"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
