from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import gradio as gr

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

def text_to_speech_gradio(text):
    # Synthesize speech from text
    speech = synthesiser(text)

    # Extract the audio array and sampling rate from the output
    audio = speech["audio"]
    sampling_rate = speech["sampling_rate"]

    # Save the audio to a WAV file
    file_path = "bark_generation.wav"
    write_wav(file_path, sampling_rate, audio)

    return file_path

# Create Gradio interface
iface = gr.Interface(
    fn=text_to_speech_gradio,
    inputs=gr.Textbox(lines=5, placeholder="Enter text here..."),
    outputs=gr.Audio(type="filepath"),
    title="Bark Text-to-Speech",
    description="Enter text and generate speech using the Bark model."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
