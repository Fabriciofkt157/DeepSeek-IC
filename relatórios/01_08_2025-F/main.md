# Descrição de um LLM (Large Language Model)
- Tarefa principal: prever a próxima palavra (ou "token") mais provável em uma sequência.
- Tudo que o LLM faz deriva da capacidade de analizar o prompt de entrada e fornecer uma saída com base na probabilidade da próxima palavra estar relacionada com o prompt.

## Pré-treinamento: 
As probabilidades são definidas a partir do Treinamento Massivo (Pré-treinamento) onde o modelo é alimentado com um conjunto de dados gigantesco representando grande parte da internet, isso inclui:  livros, artigos, códigos-fonte (do GitHub, Stack Overflow, etc.), conversas, entre outros.
- Durante esse treinamento, o modelo aprende padrões (palavras que costumam aparecer juntas ou estarem relacionadas). Ele não memoriza as informações e as apresenta para o usuário, por isso o modelo pode alucinar e fornecer informações incorretas. O aprendizado consiste em internalizar as relações estatísticas entre as palavras, assim aprendendo grámatica, sintaxe, semântica, estilos de escrita, estruturas de código e até mesmo a "aparência" de um raciocínio lógico.

## Arquitetura de LLMs atuais: Transformer
Os modelos atuais deixaram de usar a arquiteura antiga para redes neurais (como a Recurrent Neural Network) e passaram a usar o Transformer (desenvolvido a partir de 2017). O principal componente do Transformer é o Mecanismo de Atenção (Attention Mechanism) — capacidade do modelo de, ao gerar uma nova palavra, "olhar para trás" em todo o texto de entrada (prompt) e decidir quais palavras são mais importantes para contextualizar a próxima. Ele pode dar mais "peso" a uma palavra no início da pergunta, mesmo que esteja gerando uma palavra no final da resposta. Isso resolve o problema de "memória" de longo prazo que modelos antigos tinham.

## Processo de Geração (inferência): 
- Tokenização: O texto (ex: "Como gerar um array em Python?") é quebrado em pedaços menores chamados tokens. Tokens podem ser palavras, partes de palavras ou símbolos ("Como", " gerar", " um", " array", " em", " Python", "?").
- Embeddings: Cada token é convertido em um vetor numérico (uma longa lista de números). Esse vetor representa o "significado" ou a posição do token em um espaço conceitual. Palavras com significados semelhantes terão vetores semelhantes.
- Processamento pelo Transformer: Esses vetores passam pelas camadas da rede neural Transformer. O mecanismo de atenção analisa as relações entre todos os tokens de entrada.
- Previsão do Próximo Token: Na saída, o modelo gera uma distribuição de probabilidade sobre todo o seu vocabulário (milhares de tokens possíveis). Ele calcula qual é o token mais provável para vir a seguir.
- Geração Autoregressiva: O token escolhido é então adicionado à sequência de entrada, e todo o processo se repete para gerar o próximo token, e assim por diante, até que uma condição de parada seja atingida (como um token de [FIM_DA_RESPOSTA]).

## O Ajuste Fino e Alinhamento (RLHF)
Um modelo pré-treinado precisa de ajustes para fornecer respostas coerentes: ele sabe muito, mas não sabe como ser útil. Para torná-lo um assistente prestativo, ele passa por um processo de alinhamento, geralmente usando Reinforcement Learning from Human Feedback (RLHF).

Como funciona (simplificadamente):
- O modelo gera várias respostas para um mesmo prompt.
- Humanos avaliadores classificam essas respostas da melhor para a pior.
- Um "modelo de recompensa" é treinado com base nessas classificações humanas para aprender o que constitui uma "boa" resposta.
  
O LLM original é então ajustado usando reinforcement learning para maximizar a pontuação dada por esse modelo de recompensa. É assim que ele aprende a ser útil, seguro e a seguir instruções.
