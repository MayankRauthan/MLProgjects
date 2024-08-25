import requests
import json
import speech_recognition as sr
import gradio as gr
import numpy as np
import io
import wave

# Set up Hugging Face API endpoint and token
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def review_answer(question, answer, user_answer):
    payload = {
        "inputs": f"Question: {question}\nModel's Answer: {answer}\nUser's Answer: {user_answer}\nPlease review the user's answer based on the given answer and question and only give it a score out of 10. On the next line, provide very short feedback in 1 line for the user in JSON format with 'score' and 'feedback' fields."
    }
    
    response = query(payload)
    return response[0]['generated_text'].strip()

def transcribe_audio(audio_data):
    if isinstance(audio_data, tuple):
        sample_rate, audio_array = audio_data
    elif isinstance(audio_data, np.ndarray):
        sample_rate = 44100  # Assume sample rate is 44100
        audio_array = audio_data
    else:
        return "Audio data is not in a supported format.", None

    byte_io = io.BytesIO()
    with wave.open(byte_io, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.astype(np.int16).tobytes())

    byte_io.seek(0)

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(byte_io) as source:
            audio = recognizer.record(source)
        user_answer = recognizer.recognize_google(audio)
        return user_answer, None
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio.", None
    except sr.RequestError as e:
        return f"Sorry, there was an error with the request: {e}", None
    except Exception as e:
        return f"Unexpected error: {e}", None

def process_question(question, answer, audio_data):
    user_answer, transcribe_error = transcribe_audio(audio_data)
    if transcribe_error:
        return "Error in transcription", transcribe_error, None

    review = review_answer(question, answer, user_answer)

    try:
        review_data = json.loads(review)
    except json.JSONDecodeError as e:
        return f"Error decoding review JSON: {e}", None, None

    score = review_data.get('score', 'No score field')
    feedback = review_data.get('feedback', 'No feedback field')

    return score, feedback, user_answer

def create_gradio_interface():
    questions = [
        {"question": "Explain the difference between a list and a tuple in Python.", "answer": "A list is mutable, meaning its contents can be changed, while a tuple is immutable and cannot be modified once created."},
        {"question": "What are Python decorators and how are they used?", "answer": "Python decorators are functions that modify the behavior of other functions or methods. They are used by placing the decorator function above the function to be decorated, preceded by the @ symbol."},
        {"question": "How does Python's garbage collection work?", "answer": "Python's garbage collection is handled by reference counting and a cyclic garbage collector to reclaim memory used by objects that are no longer in use."}
    ]

    total_score = []

    def gradio_function(question_index, audio_data):
        question = questions[question_index]['question']
        answer = questions[question_index]['answer']
        score, feedback, user_answer = process_question(question, answer, audio_data)

        if isinstance(score, int):
            total_score.append(score)

        if question_index == len(questions) - 1:
            average_score = sum(total_score) / len(total_score) if total_score else 0
            result = "Pass" if average_score >= 7 else "Fail"
            return f"**Your Answer:** {user_answer}\n\n**Score:** {score}/10\n**Feedback:** {feedback}\n\n**Final Result:** {result} (Average Score: {average_score:.2f}/10)"
        else:
            return f"**Your Answer:** {user_answer}\n\n**Score:** {score}/10\n**Feedback:** {feedback}"

    with gr.Blocks() as demo:
        gr.Markdown("### Answer the following questions:")
        for i, q in enumerate(questions):
            with gr.Row():
                gr.Markdown(f"**Question {i + 1}:** {q['question']}")
                audio_input = gr.Audio(type="numpy", label=f"Record your answer for question {i + 1}")
                output_text = gr.Textbox(label="Review", lines=5)
                audio_input.change(gradio_function, inputs=[gr.Slider(minimum=0, maximum=len(questions)-1, value=i, visible=False), audio_input], outputs=output_text)

    demo.launch()

if __name__ == "__main__":
    create_gradio_interface()
