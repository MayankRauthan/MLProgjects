import streamlit as st
from RealtimeSTT import AudioToTextRecorder

def main():
    st.title("Real-time Speech-to-Text")
    st.write("Say something and see the text appear in real-time!")
    st.write("Listenings...")
    # Placeholder for the real-time text
    text_placeholder = st.empty()

    # Create an instance of AudioToTextRecorder
    with AudioToTextRecorder(spinner=False, model="tiny.en", language="en") as recorder:
        

        # Continuously update the text in the Streamlit app
        while True:
            text_placeholder.text(recorder.text())

if __name__ == '__main__':
    main()
