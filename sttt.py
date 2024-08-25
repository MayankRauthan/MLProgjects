from RealtimeSTT import AudioToTextRecorder
if __name__ == '__main__':
    recorder = AudioToTextRecorder(spinner=False, model="small", language="hi",enable_realtime_transcription=True)

    print("Say something...")
    while (True): print(recorder.text(), end=" ", flush=True)