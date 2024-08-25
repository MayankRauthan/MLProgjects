import os
import shutil
import torch
import time
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

def answer_question(question: str, context: str, max_chunk_length: int = 510) -> str:
    answers = []
    chunks = [context[i:i + max_chunk_length] for i in range(0, len(context), max_chunk_length)]

    for chunk in chunks:
        inputs = tokenizer.encode_plus(question, chunk, add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"].tolist()[0]

        # Perform the question answering
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        # Get the most likely answer
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        answer = answer.replace("[SEP]", "").strip()
        answers.append(answer)

    # Combine answers from all chunks
    final_answer = " ".join(answers)
    return final_answer

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

# Initialize context
context = ""

@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Question Answering</title>
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
            .upload-container, .result-container {
                margin-bottom: 24px;
            }
            .upload-label, .result-label {
                display: block;
                margin-bottom: 8px;
                font-size: 18px;
            }
            #contextInput, #questionInput {
                width: 100%;
                padding: 12px;
                border: 1px solid #ffffff;
                border-radius: 4px;
                background-color: #1e1e2e;
                color: #ffffff;
                margin-bottom: 12px;
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
            #answerResult, #processTime, #wordCount {
                width: 100%;
                padding: 12px;
                border: 1px solid #ffffff;
                border-radius: 4px;
                background-color: #1e1e2e;
                color: #ffffff;
                margin-bottom: 12px;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Question Answering</h1>
            <div class="upload-container">
                <label for="contextInput" class="upload-label">Upload Context File</label>
                <input type="file" id="contextInput" accept=".txt">
                <label for="questionInput" class="upload-label">Ask a Question</label>
                <input type="text" id="questionInput" placeholder="Enter your question">
            </div>
            <div class="buttons">
                <button id="clearButton">Clear</button>
                <button id="submitButton">Get Answer</button>
            </div>
            <div class="result-container">
                <label for="answerResult" class="result-label">Answer</label>
                <textarea id="answerResult" rows="10" readonly></textarea>
                <label for="processTime" class="result-label">Process Time (seconds)</label>
                <textarea id="processTime" rows="1" readonly></textarea>
                <label for="wordCount" class="result-label">Word Count</label>
                <textarea id="wordCount" rows="1" readonly></textarea>
            </div>
        </div>
        <script>
            document.getElementById('submitButton').addEventListener('click', async () => {
                const contextFile = document.getElementById('contextInput').files[0];
                const question = document.getElementById('questionInput').value;

                if (!contextFile || !question) {
                    alert('Please upload a context file and enter a question.');
                    return;
                }

                const formData = new FormData();
                formData.append('file', contextFile);
                formData.append('question', question);

                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                document.getElementById('answerResult').textContent = data.answer || 'No answer found.';
                document.getElementById('processTime').textContent = data.process_time || 'N/A';
                document.getElementById('wordCount').textContent = data.word_count || 'N/A';
            });

            document.getElementById('clearButton').addEventListener('click', () => {
                document.getElementById('contextInput').value = '';
                document.getElementById('questionInput').value = '';
                document.getElementById('answerResult').textContent = '';
                document.getElementById('processTime').textContent = '';
                document.getElementById('wordCount').textContent = '';
            });
        </script>
    </body>
    </html>
    """

@app.post("/upload-video/", response_class=JSONResponse)
async def upload_file(file: UploadFile = File(...), question: str = Form(...)):
    global context
    try:
        file_location = f"{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(file_location, "r", encoding="utf-8") as f:
            context = f.read()

        start_time = time.time()
        answer = answer_question(question, context)
        end_time = time.time()
        process_time = end_time - start_time
        word_count = len(answer.split())

        os.remove(file_location)
        return {"answer": answer, "process_time": process_time, "word_count": word_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
