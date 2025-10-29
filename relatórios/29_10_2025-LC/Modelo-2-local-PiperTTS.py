#Text to speech translation with Piper TTS and python
# by TechMakerAi on youtube 

from piper import pipervoice
import wave

from io import BytesIO
from pygame import mixer 

mixer.init() #inicializar o mixer da biblioteca Pygame

mp3audio = BytesIO() #guardar o arquivo de audio na memória

voice = Piper.Voice.Load(#'aqui fica o caminho onde está instalada a voz do pacote Piper',
    config_path=#"o mesmo caminho para o arquivo porém .json")


text = #'nessa variável a resposta fornecida pelo LLM implementado'

with wave.open(mp3audio, "wb") as wave_file: #tradução do texto para formato de onda (PIPER)
    voice.synthesize(text, wav_file)
    wav_file.close()

mp3audio.seek(0) #garantindo que o audio toque do início corretamente

mixer.music.load(mp3audio, "wav") #tocando o audio 
mier.music.play()

while music.mixer.get_busy():
    pass 

