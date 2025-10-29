# Estratégias para a implementação da fala
Para implementação da fala com raspberry encontrei várias ferramentas, dentre elas as mais práticas e viáveis foram as do canal TechMakerAI no youtube.
Sendo elas as mais diretas, suas implementações são simples necessitando apenas de algumas adaptações no quesito de captação de texto.
## Modelo número 1
Implementaçao da API gratuita do Google GTTS
[Modelo googleTTS](
Modelo-1-online-googleTTS.py)
### Vantagens 
- Mais leve
- Compatível com todas as alternativas de geração de texto
- API estável
### Desvantagens
- Necessita de conexão com a internet

## Modelo número 2
Implementação do *Piper* uma engine de text-to-speech local 
[Modelo PiperTTS](Modelo-2-local-PiperTTS.py)
### Vantagens
- Local
- Altamente customizável: voz, velocidade da fala, qualidade do audio, etc...
- Implementação simples
- Alta compatibilidade visto que foi projetado inicialmente para o *Raspberry pi*
### Desvantagens 
- Talvez seja levemente mais pesado impossibilitando a implementação de alguma alternativa de LLM
- Talvez Necessitar otimizações

