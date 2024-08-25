import os
import google.generativeai as genai
import json
import gradio as gr
import speech_recognition as sr

# Set up API key
genai.configure(api_key="AIzaSyAkJZU8hwbSKITJ12uh-piT9MeJBO0DS1g")  # Replace with your actual API key

# Define the generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

# Create the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

def generate_question(topic, jobrole, num_ques):
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    f"Topic: {topic}, Please generate {num_ques} interview questions and answers in json format based on this topic and {jobrole}\n.Please generate question and answer as field in json format.",
                ],
            },
        ]
    )
    response = chat_session.send_message("INSERT_INPUT_HERE")
    return response.text.strip()

def review_answer(question, answer, user_answer):
    review_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    f"Question: {question}\n"
                    f"Model's Answer: {answer}\n"
                    f"User's Answer: {user_answer}\n"
                    "Please review the user's answer based on given answer and question and only give it a score out of 10 and just give the score and on next line only provide very short feedback in 1 line for user in a way like you are talking to user.Please give score and feedback as separate field in json format"
                ],
            },
        ]
    )
    review_response = review_session.send_message("INSERT_INPUT_HERE")
    return review_response.text.strip()

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

def process_interview(topic, jobrole, num_ques):
    question_response = generate_question(topic, jobrole, num_ques)
    modified_json_string = question_response.replace('json', '').replace('```', '')
    
    try:
        data = json.loads(modified_json_string)
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"
    
    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]
    
    return questions, answers

def gradio_interface():
    with gr.Blocks() as demo:
        topic = gr.Textbox(label="Select a topic:")
        jobrole = gr.Textbox(label="Select job role:")
        num_ques = gr.Number(label="Number of Questions:", value=3)
        generate_button = gr.Button("Generate Questions")
        
        questions_output = gr.Textbox(label="Generated Questions", interactive=False)
        current_question = gr.Textbox(label="Current Question", interactive=False)
        
        with gr.Row():
            start_recording = gr.Button("Start Recording")
            stop_recording = gr.Button("Stop Recording")
        
        user_answer = gr.Textbox(label="Your Answer (Spoken)", interactive=False)
        submit_answer = gr.Button("Submit Answer")
        
        feedback = gr.Textbox(label="Feedback", interactive=False)
        overall_score = gr.Textbox(label="Overall Score", interactive=False)
        
        questions_state = gr.State([])
        answers_state = gr.State([])
        current_question_index = gr.State(0)
        user_answers_state = gr.State([])
        
        def generate_questions(topic, jobrole, num_ques):
            questions, answers = process_interview(topic, jobrole, int(num_ques))
            return {
                questions_output: "\n".join(questions),
                current_question: questions[0] if questions else "",
                questions_state: questions,
                answers_state: answers,
                current_question_index: 0,
                user_answers_state: []
            }
        
        def record_answer():
            return speech_to_text()
        
        def submit_current_answer(questions, answers, current_index, user_answers, spoken_answer):
            if current_index < len(questions):
                review = review_answer(questions[current_index], answers[current_index], spoken_answer)
                review_data = json.loads(review.replace('json', '').replace('```', ''))
                
                user_answers.append(spoken_answer)
                next_index = current_index + 1
                next_question = questions[next_index] if next_index < len(questions) else "Interview Complete"
                
                feedback_text = f"Question: {questions[current_index]}\nYour Answer: {spoken_answer}\nScore: {review_data['score']}/10\nFeedback: {review_data['feedback']}"
                
                if next_index == len(questions):
                    total_score = sum(int(json.loads(review_answer(q, a, ua).replace('json', '').replace('```', ''))['score']) for q, a, ua in zip(questions, answers, user_answers))
                    avg_score = total_score / len(questions)
                    overall_score_text = f"Your Overall Score: {avg_score:.2f}/10"
                else:
                    overall_score_text = ""
                
                return {
                    current_question: next_question,
                    feedback: feedback_text,
                    overall_score: overall_score_text,
                    current_question_index: next_index,
                    user_answers_state: user_answers,
                    user_answer: ""
                }
            else:
                return {current_question: "Interview Complete"}
        
        generate_button.click(generate_questions, inputs=[topic, jobrole, num_ques], 
                              outputs=[questions_output, current_question, questions_state, answers_state, current_question_index, user_answers_state])
        
        start_recording.click(record_answer, outputs=user_answer)
        stop_recording.click(lambda: None)  # Placeholder for stop recording functionality
        
        submit_answer.click(submit_current_answer, 
                            inputs=[questions_state, answers_state, current_question_index, user_answers_state, user_answer],
                            outputs=[current_question, feedback, overall_score, current_question_index, user_answers_state, user_answer])

    demo.launch(share=True)

if __name__ == "__main__":
    gradio_interface()