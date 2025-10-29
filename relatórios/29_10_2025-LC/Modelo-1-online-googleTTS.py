# Text to speech translation with google TTS and python
# by TechMakerAI on Youtube

from gtts import gTTS
import os 
from io import BytesIO
from pygame import mixer 

mixer.init()

mp3audio = BytesIO() #salvado espaço para arquivo de áudio na memória

text = #'nessa variável a resposta fornecida pelo LLM implementado'

tts = gTTS(text, lang = 'pt', tld = 'com.br')# comunicação com a API do google TTS

tts.write_to_fp(mp3audio) #

mp3audio.seek(0) #sincronizar para o começo do arquivo de audio para total audibilidade do texto enviado

mixer.music.load(mp3audio, "mp3")
mixer.music.play() #tocando o arquivo de aúdio com o pygame

while mixer.music.get_busy():
    pass
