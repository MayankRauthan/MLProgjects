import RealtimeSTT
import time

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    def on_start():
        print("listening")
    def on_stop():
        print("transcribing")

    # Create a recorder instance
    from RealtimeSTT import AudioToTextRecorder
    
# Create a recorder instance
    with AudioToTextRecorder(spinner=False, model="tiny.en",on_recording_start=on_start,on_recording_stop=on_stop) as recorder:
         print(recorder.text())
    # Print the transcribed text
