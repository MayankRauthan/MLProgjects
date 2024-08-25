import gradio as gr
import speech_recognition as sr

# Initialize the recognizer and microphone
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def recognize_speech():
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio_data = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand the audio"
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition service"

# Create Gradio interface
interface = gr.Interface(
    fn=recognize_speech,
    inputs="microphone",
    outputs="text",
    live=True,
    title="Live Speech-to-Text"
)

# Launch the app
interface.launch()
