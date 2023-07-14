import vosk
import tts
import pvporcupine
from pvrecorder import PvRecorder
import time
import struct

key = 'Eyg9RqzHDBvGcBS9AbyVVTZwPMcIfOXyAT4LX4+JvQTRVUvhTSIDdQ=='
porcupine = pvporcupine.create(
    access_key=key,
    keywords=['jarvis'],
    sensitivities=[1]
)

model = vosk.Model("model_small")
samplerate = 16000
device = -1
kaldi_rec = vosk.KaldiRecognizer(model, samplerate)
# q = queue.Queue()

recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)

recorder.start()
print('Using device: %s' % recorder.selected_device)

print(f"Jarvis (v1.0) начал свою работу ...")
tts.va_speak('я готов.')
time.sleep(0.5)

ltc = time.time()

while True:
    try:
        pcm = recorder.read()
        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            recorder.stop()
            tts.va_speak("Да, сэр.")
            print("Yes, sir.")
            recorder.start()  # prevent self recording
            ltc = time.time()

        while time.time() - ltc <= 5:
            pcm = recorder.read()
            sp = struct.pack("h" * len(pcm), *pcm)

            if kaldi_rec.AcceptWaveform(sp):
                print(kaldi_rec.FinalResult())
                lts = time.time()
                break

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise