# Relatório — RNA Multicamadas para Sudoku 4x4

## 1. Introdução

O objetivo da atividade é propor uma solução com Rede Neural Artificial multicamadas para resolver o Sudoku 4x4, composto por subgrupos 2x2 e valores pertencentes ao conjunto S = {1, 2, 3, 4}. O problema foi tratado como uma tarefa de aprendizado supervisionado, em que a entrada é um tabuleiro parcialmente preenchido e a saída esperada é o tabuleiro completo válido.

Embora o Sudoku possa ser apresentado como um jogo, do ponto de vista da Inteligência Artificial ele é principalmente um problema de satisfação de restrições. Cada solução precisa obedecer simultaneamente a regras sobre células, linhas, colunas e blocos. Por isso, a solução proposta combina aprendizado neural com validação simbólica.

## 2. Representação do problema

Cada tabuleiro 4x4 possui 16 células. Uma célula vazia é representada pelo valor 0. Os valores possíveis para preenchimento são 1, 2, 3 e 4.

A entrada da RNA é o tabuleiro incompleto codificado por one-hot encoding. Cada célula possui cinco possibilidades: vazio, 1, 2, 3 ou 4. Assim, o vetor de entrada possui 16 × 5 = 80 posições.

A saída da rede possui 16 posições, uma para cada célula. Para cada posição, a rede prevê uma das quatro classes possíveis, correspondentes aos números 1, 2, 3 e 4.

## 3. Geração dos dados

O conjunto de dados foi gerado artificialmente. Primeiro, foram produzidos tabuleiros completos válidos de Sudoku 4x4. Em seguida, algumas posições foram removidas aleatoriamente, criando tabuleiros iniciais com células vazias.

Dessa forma, cada amostra possui:

- entrada: tabuleiro incompleto;
- saída: tabuleiro completo original.

Essa estratégia permite gerar dados de treino e teste controlados, garantindo que todas as soluções usadas sejam válidas.

## 4. Arquitetura da RNA

A rede usada é uma MLP com camadas densas. A arquitetura geral é:

```text
Entrada: 80 neurônios
Camada oculta 1: 128 neurônios com ReLU
Camada oculta 2: 128 neurônios com ReLU
Camada oculta 3: 64 neurônios com ReLU
Saída: 16 × 4 neurônios
```

A função de perda utilizada é a entropia cruzada, pois cada célula é tratada como uma classificação entre quatro valores possíveis.

## 5. Validação simbólica

Após a previsão da RNA, o sistema verifica se o tabuleiro respeita as regras do Sudoku. A função de validação confere:

- se cada linha contém exatamente {1, 2, 3, 4};
- se cada coluna contém exatamente {1, 2, 3, 4};
- se cada bloco 2x2 contém exatamente {1, 2, 3, 4}.

Como a RNA pode prever valores repetidos, foi implementado um backtracking guiado pelas probabilidades produzidas pela rede. Assim, a rede indica quais valores parecem mais prováveis, e o raciocínio simbólico garante que a solução final seja válida.

## 6. Dificuldade em gerar amostras

O problema de gerar amostras está relacionado à explosão combinatória. Para Sudoku 4x4, é possível gerar todos os tabuleiros válidos e criar exemplos de treinamento. Entretanto, ao generalizar para NxN, o número de combinações cresce rapidamente.

No Sudoku 9x9, por exemplo, existe uma quantidade extremamente grande de grades completas possíveis. Isso torna inviável gerar e testar todas as combinações por força bruta. Portanto, o problema deixa de ser apenas uma tarefa de classificação e passa a exigir raciocínio sobre restrições.

## 7. Por que é um problema de raciocínio?

O Sudoku não depende apenas de reconhecimento de padrões. Para completar corretamente uma célula, é necessário considerar simultaneamente a linha, a coluna e o bloco ao qual ela pertence. Isso caracteriza um problema de raciocínio lógico e satisfação de restrições.

A RNA consegue aprender regularidades estatísticas a partir dos exemplos, mas não garante, sozinha, que todas as restrições sejam atendidas. Por isso, a solução proposta utiliza uma abordagem híbrida: aprendizado neural para sugerir valores e raciocínio simbólico para validar e corrigir a solução.

## 8. Generalização de 4x4 para NxN

A generalização direta de uma RNA treinada em Sudoku 4x4 para Sudoku NxN não é simples. A quantidade de células aumenta, o número de valores possíveis aumenta e a estrutura dos blocos muda. Portanto, a arquitetura da rede e a representação da entrada precisariam ser adaptadas.

Além disso, quanto maior o Sudoku, maior a dificuldade de gerar amostras suficientes para cobrir o espaço de possibilidades. Isso reforça a importância de integrar aprendizado de máquina com métodos simbólicos, como SAT, CSP, backtracking e regras lógicas.

## 9. Conclusão

A solução proposta demonstra que uma RNA multicamadas pode ser usada para aprender padrões de preenchimento em Sudoku 4x4. Porém, como o Sudoku é essencialmente um problema de restrições, a rede neural sozinha não é suficiente para garantir validade lógica em todos os casos.

A integração entre a RNA e o validador simbólico produz uma solução mais robusta, pois combina aprendizado estatístico com raciocínio lógico. Essa abordagem se aproxima da IA neurossimbólica, em que modelos de aprendizado são combinados com representação simbólica para obter maior confiabilidade e explicabilidade.
