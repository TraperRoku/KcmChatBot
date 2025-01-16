import torch, transformers, numpy as np, librosa, pvrecorder
import logging, warnings, pathlib, sys
import logging, warnings, pathlib
import torch, transformers
import time
import sounddevice
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.getLogger('transformers').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


class TextToSpeech:
    def __init__(self):
        preload_models()

    def __call__(self, text: str):
        audio_array = generate_audio(text)

        sounddevice.play(audio_array, SAMPLE_RATE)
        time.sleep(len(audio_array) / SAMPLE_RATE) 

class SpeechToText:
    def __init__(self, cache_dir: pathlib.Path = pathlib.Path("cache-dir")):

        cache_dir.mkdir(exist_ok=True, parents=True)

        # Źródło modelu:
        # https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english

        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

        self.proc = transformers.Wav2Vec2Processor.from_pretrained(
            model_name, cache_dir=cache_dir)
        self.model = transformers.Wav2Vec2ForCTC.from_pretrained(
            model_name, cache_dir=cache_dir)

    def __call__(self, speech: np.ndarray) -> str:
        input_model = self.proc(speech, sampling_rate=16000,
                                return_tensors="pt", padding=True)

        output_model = torch.argmax(
            self.model(
                input_model.input_values,
                attention_mask=input_model.attention_mask).logits, dim=-1)

        return self.proc.batch_decode(output_model)[0]

class Chatboot:
    def __init__(self, cache_dir: Path = Path("cache-dir")):

        cache_dir.mkdir(exist_ok=True, parents=True)

        # Źródło modelu:
        #https://huggingface.co/microsoft/DialoGPT-medium

        model_name = "microsoft/DialoGPT-medium"

        self.proc = AutoTokenizer.from_pretrained(model_name,
                    cache_dir=cache_dir, padding_size='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                     cache_dir=cache_dir)


    def __call__(self, question: str):
        q = self.proc.encode(question + self.proc.eos_token, 
                             return_tensors='pt')

        output = self.model.generate(q, max_length=1000, 
                                      pad_token_id=self.proc.eos_token_id)
        answer = self.proc.decode(output[:, q.shape[-1]:][0], 
                                  skip_special_tokens=True)
        
        return answer

if __name__ == "__main__":
    chatboot = Chatboot()
    speech2text = SpeechToText()
    text2speech = TextToSpeech()
    audio_files = pathlib.Path("audio-files")

    print("\nPrzykład rozpoznawania mowy z plików audio:")
    for i, file in enumerate([file for file in audio_files.glob("*.wav")]):
        samples, _ = librosa.load(file, sr=16000)
        text = speech2text(samples)
        print(f"    {file.stem}: {text}")


    # Sprawdzenie dostępnych urządzeń nagrywających
    # -------------------------------------------------------------------------
    print("\nDostępne urządzenia nagrywające:")
    devices = pvrecorder.PvRecorder.get_available_devices()
    for index, device in enumerate(devices):
        print(f"    [{index}] {device}")

    
    # Przykład przechwytywania sygnału z mikrofonu dla biblioteki pvrecorder
    # -------------------------------------------------------------------------
    
    # Wybór urządzenia nagrywającego
    device_id = 0     

    # Ustawienie długość nagrywanej ramki sygnału mowy
    sample_rate = 16000
    frame_length = 512 
    time2 = 5
    n_frames = (time2 * sample_rate) // frame_length
    
    # Inicjalizacja obiektu PvRecorder: sampling_rate=16kHz
    recorder = pvrecorder.PvRecorder(device_index=device_id, 
                                     frame_length=frame_length)

    # Przygotowanie listy do gromadzenia nagrywanych ramek sygnału
    while True :
        recording = []      

    # Rozpoczęcie nagrywania
        input("\nNaciśnij ENTER by rozpocząć nagrywanie (5s):")
        recorder.start()  

    # Zapis nagrywanych danych
        for _ in range(n_frames):
            recording.extend(recorder.read())  

    # Zakończenie nagrywania
        print("Nagrywanie: STOP")
        recorder.stop()     

    # Normalizacja zarejestrowanego sygnału
        recording /= np.max(np.abs(recording))

    # Wyświetlenie treści nagranego sygnału
        print(speech2text(recording))

        question = speech2text(recording)
        answer = chatboot(question)

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}\n")
        
        text2speech(answer)

    
