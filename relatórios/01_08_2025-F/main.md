# Descrição de um LLM (Large Language Model)
- Tarefa principal: prever a próxima palavra (ou "token") mais provável em uma sequência.
- Tudo que o LLM faz deriva da capacidade de analizar o prompt de entrada e fornecer uma saída com base na probabilidade da próxima palavra estar relacionada com o prompt.

## Pré-treinamento: 
As probabilidades são definidas a partir do Treinamento Massivo (Pré-treinamento) onde o modelo é alimentado com um conjunto de dados gigantesco representando grande parte da internet, isso inclui:  livros, artigos, códigos-fonte (do GitHub, Stack Overflow, etc.), conversas, entre outros.
- Durante esse treinamento, o modelo aprende padrões (palavras que costumam aparecer juntas ou estarem relacionadas). Ele não memoriza as informações e as apresenta para o usuário, por isso o modelo pode alucinar e fornecer informações incorretas. O aprendizado consiste em internalizar as relações estatísticas entre as palavras, assim aprendendo grámatica, sintaxe, semântica, estilos de escrita, estruturas de código e até mesmo a "aparência" de um raciocínio lógico.

## Arquitetura de LLMs atuais: Transformer
