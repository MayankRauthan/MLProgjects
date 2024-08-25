import gradio as gr
import speech_recognition as sr

def recognize_speech():
    def inner_recognize_speech(person):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print(f"{person}, please wait.")
            recognizer.adjust_for_ambient_noise(source)
            print(f"{person}, say something!")
            audio_data = recognizer.listen(source)
            print("Recognizing...")
            try:
                text = recognizer.recognize_google(audio_data)
                return f"{person} said: " + text
            except sr.UnknownValueError:
                return f"{person} said: Google Speech Recognition could not understand the audio"
            except sr.RequestError:
                return f"{person} said: Could not request results from Google Speech Recognition service"
    
    # Call the inner function for different persons
    text_person1 = inner_recognize_speech("Person 1")
    text_person2 = inner_recognize_speech("Person 2")
    
    # Return a tuple of outputs for Gradio to handle
    return (text_person1, text_person2)

# Create Gradio interface
interface = gr.Interface(
    fn=recognize_speech,
    inputs=None,
    outputs=["text", "text"],
    live=True,
    title="Speech Recognition App",
    description="Separate speech recognition for Person 1 and Person 2"
)

# Launch the app
interface.launch()
