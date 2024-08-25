from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import tempfile
import librosa
import numpy as np
from transformers import pipeline
from pyannote.audio import Pipeline
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load the ASR and diarization pipelines
try:
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=device
    )
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0",
                                                    use_auth_token="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")
except Exception as e:
    raise RuntimeError(f"Error initializing pipelines: {e}")

# Function to transcribe audio segments
def transcribe_segment(segment_audio, sr):
    # Convert the segment to the required format
    segment_audio = librosa.resample(segment_audio, orig_sr=sr, target_sr=16000)
    segment_audio = segment_audio.astype(np.float32)

    # Perform ASR
    result = asr_pipeline(segment_audio)
    return result["text"]

def transcribe_and_diarize(audio_file_path, speaker_names):
    # Load audio file
    audio, sr = librosa.load(audio_file_path, sr=None)

    # Run diarization
    diarization = diarization_pipeline(audio_file_path)

    # Create a mapping of speaker segments
    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append((turn.start, turn.end))

    # Transcribe each segment
    transcripts = []
    word_counts = {}
    for speaker, segments in speaker_segments.items():
        speaker_transcript = []
        word_count = 0
        for start, end in segments:
            segment_audio = audio[int(start * sr):int(end * sr)]
            segment_transcript = transcribe_segment(segment_audio, sr)
            word_count += len(segment_transcript.split())
            speaker_transcript.append((start, end, speaker, segment_transcript))
        transcripts.extend(speaker_transcript)
        word_counts[speaker] = word_count

    # Filter transcripts
    filtered_transcripts = [(start, end, speaker, text) for (start, end, speaker, text) in transcripts if text.strip()]

    # Sort transcripts by start time
    sorted_transcripts = sorted(filtered_transcripts, key=lambda x: x[0])

    # Map speakers to provided names
    speaker_map = {f"SPEAKER_{i:02d}": speaker_names[i] if i < len(speaker_names) else f"Speaker {i + 1}" for i in
                   range(len(speaker_names))}

    # Format results for display
    result = ""
    for start, end, speaker, text in sorted_transcripts:
        speaker_name = speaker_map.get(speaker, speaker)
        result += f"{start:.2f}-{end:.2f} {speaker_name}: {text}\n"

    # Append word counts to result
    result += "\nWord Counts:\n"
    for speaker, count in word_counts.items():
        speaker_name = speaker_map.get(speaker, speaker)
        result += f"{speaker_name}: {count} words\n"

    return result

# Create FastAPI app
app = FastAPI()

# Serve static files from the directory
app.mount("/static", StaticFiles(directory=r"H:\AppData\Ml projects\static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r") as file:
        return HTMLResponse(content=file.read())

@app.get("/diarization", response_class=HTMLResponse)
async def serve_diarization_page():
    with open("static/diarization.html", "r") as file:
        return HTMLResponse(content=file.read())

@app.post("/transcribe_diarize", response_class=HTMLResponse)
async def transcribe_diarize(request: Request, audio_file: UploadFile = File(...), speaker_names: str = Form(...)):
    try:
        # Save the uploaded audio file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            audio_file_path = tmp_file.name
            with open(audio_file_path, "wb") as f:
                f.write(await audio_file.read())

        # Process the audio file
        speaker_names_list = [name.strip() for name in speaker_names.split(",")]
        result = transcribe_and_diarize(audio_file_path, speaker_names_list)

        # Delete the temporary audio file
        os.remove(audio_file_path)

        return HTMLResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_message = f"Error: {str(e)}"
        return HTMLResponse(content=f"<h2>{error_message}</h2>", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

