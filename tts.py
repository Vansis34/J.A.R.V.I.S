import time
import os
import sounddevice as sd
import torch
from num2words import num2words
import re

language = 'ru'
model_id = 'v3_1_ru'
sample_rate = 24000  # 48000
speaker = 'eugene'  # aidar, baya, kseniya, xenia, random
put_accent = True
put_yo = True
device = torch.device('cpu')  # cpu или gpu

#  Для ипользования модели оффлайн
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                   local_file) 

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")

#  Для использования модели онлайн
# model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
#                           model='silero_tts',
#                           language=language,
#                           speaker=model_id)
model.to(device)

def _nums_to_text(text) -> str:
        """Преобразует числа в буквы: 1 -> один, 23 -> двадцать три"""
        return re.sub(
            r"(\d+)",
            lambda x: num2words(int(x.group(0)), lang=language),
            text,
        )

# воспроизводим
def va_speak(what: str):
    normt = _nums_to_text(what)
    audio = model.apply_tts(text=normt + "..",
                            speaker=speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo)

    sd.play(audio, sample_rate * 0.95)
    time.sleep((len(audio) / sample_rate) + 0.5)
    sd.stop()

#  Используй функцию ниже для озвучки. Не озвучивает цифры. (Алерт)
if __name__ == "__main__":
    va_speak("Выдры в недрах сушат кедры")
