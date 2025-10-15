# Possibilidades para nossa LLM:
Criar um modelo do tamanho do Gemini, DeepSeek, ChatGPT, etc, do zero é inviável (requer milhões em hardware e datasets gigantescos). A seguir alguns caminhos mais "pés no chão":

## Alternativa 1: Usar APIs de Modelos Existentes
Este é o ponto de partida mais comum e poderoso. Consiste em usar um modelo de ponta (como o DeepSeek, Gemini, etc) como um serviço (note que nem sempre é necessário pagar por esse serviço, algumas distribuições do Gemini possuem cotas gratuitas). 
- Como: Nós fazemos chamadas de API para o modelo, enviando prompts obtidos na captação (projeto da Thalita) e recebendo as respostas em formato JSON.
- Vantagens: Acesso imediato a um dos modelos mais poderosos do mundo, sem se preocupar com infraestrutura, treinamento ou custos de manutenção.
- Desvantagens: Essa abordagem requer conexão com internet, o que pode não ser o ideal em feiras de exposição, por exemplo.

Uma abordagem totalmente viável que contorna a necessidade da infraestrutura online é baixar uma distribuição do modelo e assim não depender mais da API na nuvem. Assim passamos para uma abordagem de auto-hospedagem local (self-hosting). Dessa vez usaremos um computador local para a geração das respostas e envia-las para o Pi responsável pela pronúncia. 

### Arquitetura do Projeto: PC como Servidor, Raspberry Pi como Cliente
A comunicação funcionará em uma rede local (Wi-Fi, cabeada, etc), sem precisar de internet.
- Raspberry Pi (Cliente): Captura o prompt > envia esse prompt via Wi-Fi para o endereço de IP do PC na rede.
- PC (Servidor): Está rodando um LLM localmente > recebe a requisição do Raspberry Pi > processa o prompt no LLM > gera a resposta em texto e envia de volta para o Raspberry Pi.
- Raspberry Pi (O Cliente de novo): Recebe o texto da resposta > Usa o projeto do Text-to-Speech (TTS) para converter esse texto em áudio e "pronunciar" pelos alto-falantes.

## Alternativa 2: Fazer Fine-Tuning de um Modelo Open Source
Este é o caminho se formos utilizar a IA especializada em uma tarefa ou domínio muito específico.

- Como: Pegamos um modelo de base pré-treinado e de código aberto (como o próprio DeepSeek ou os da família Llama, Mistral ou Gemma do Google) e continua o treinamento dele com nosso próprio conjunto de dados. Por exemplo, "ensinamos" a ele quem ele é e as perguntas mais comuns. Como não precisamos de uma resposta extremamente precisa (seria o ideal, mas custa processamento), podemos usar mecanismos como 

Vantagens: Alta especialização, mais controle sobre o comportamento do modelo e potencial para custos de inferência mais baixos (se você hospedar o modelo).

Plataformas: Hugging Face é o principal hub para isso. Ele oferece modelos, datasets e ferramentas (como a biblioteca transformers) para facilitar o fine-tuning.
