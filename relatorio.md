# Relatório – Resolução de Sudoku 4x4 com Rede Neural Artificial Multicamadas

## 1. Introdução

O Sudoku é um problema clássico de satisfação de restrições (Constraint Satisfaction Problem – CSP), amplamente utilizado em pesquisas de Inteligência Artificial. O objetivo consiste em preencher uma grade de forma que cada linha, coluna e bloco contenham todos os valores permitidos sem repetições.

Embora possa ser tratado como um jogo de lógica, o Sudoku representa um problema de raciocínio simbólico, pois a determinação de cada valor depende da análise simultânea de múltiplas restrições. Dessa forma, o problema pode ser modelado utilizando lógica proposicional, satisfatibilidade booleana (SAT), técnicas de busca, heurísticas e métodos de aprendizado de máquina.

Neste trabalho foi desenvolvida uma solução baseada em Rede Neural Artificial Multicamadas (MLP) para resolver instâncias de Sudoku 4x4. Além do aprendizado supervisionado realizado pela rede neural, foi utilizado um mecanismo de validação simbólica baseado nas regras do jogo, permitindo a correção de soluções inválidas por meio de backtracking guiado pelas probabilidades produzidas pela rede.

## 2. Fundamentação Teórica

### 2.1 Sudoku como Problema de Satisfação de Restrições

O Sudoku pode ser representado como um CSP, no qual cada célula corresponde a uma variável e cada variável deve assumir um valor pertencente ao conjunto:

S = {1, 2, 3, 4}

No Sudoku 4x4, a solução deve obedecer simultaneamente às seguintes restrições:

- Cada célula deve conter exatamente um valor.
- Cada linha deve conter todos os elementos de S sem repetição.
- Cada coluna deve conter todos os elementos de S sem repetição.
- Cada bloco 2x2 deve conter todos os elementos de S sem repetição.

Essas restrições também podem ser expressas por meio de lógica proposicional e utilizadas por solucionadores SAT ou sistemas de inferência lógica.

### 2.2 Redes Neurais Multicamadas

As Redes Neurais Artificiais Multicamadas, também chamadas de Multilayer Perceptron (MLP), são modelos computacionais formados por uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída.

Durante o treinamento, a rede ajusta seus pesos internos para minimizar uma função de erro, permitindo aprender padrões presentes nos dados de treinamento.

Neste trabalho, a RNA foi utilizada para aprender a relação entre tabuleiros incompletos e suas respectivas soluções completas.

### 2.3 IA Neurosimbólica

A IA neurossimbólica busca integrar técnicas de aprendizado estatístico com mecanismos simbólicos de raciocínio.

Enquanto as redes neurais apresentam grande capacidade de aprendizado a partir de exemplos, sistemas simbólicos permitem representar conhecimento de forma explícita e verificável.

A solução desenvolvida segue essa filosofia ao combinar:

- RNA para previsão dos valores;
- Validação lógica das restrições;
- Backtracking para correção de inconsistências.

## 3. Representação do Problema

Cada tabuleiro Sudoku 4x4 possui 16 células. As células vazias são representadas pelo valor 0.

Para alimentar a rede neural, foi utilizada codificação one-hot encoding. Cada célula pode assumir cinco estados possíveis:

- vazio;
- 1;
- 2;
- 3;
- 4.

Assim, cada célula é representada por um vetor de cinco posições. Como existem 16 células, a entrada final da rede possui:

16 × 5 = 80 atributos.

A saída consiste na previsão dos valores corretos para as 16 células do tabuleiro. Para cada célula, a rede escolhe uma das quatro classes possíveis: 1, 2, 3 ou 4.

## 4. Geração dos Dados

O conjunto de dados foi gerado artificialmente. Inicialmente foram produzidos tabuleiros completos válidos de Sudoku 4x4. Posteriormente, algumas posições foram removidas aleatoriamente para criar tabuleiros incompletos.

Cada amostra contém:

- Entrada: tabuleiro incompleto;
- Saída: tabuleiro completo correspondente.

Essa estratégia garante que todas as soluções utilizadas durante o treinamento sejam válidas e respeitem as restrições do problema.

Além disso, a remoção aleatória de pistas produz diferentes configurações de entrada, permitindo que a rede aprenda padrões mais variados.

## 5. Arquitetura da Rede Neural

A arquitetura implementada foi composta por:

```text
Entrada: 80 neurônios
Camada oculta 1: 128 neurônios com função ReLU
Camada oculta 2: 128 neurônios com função ReLU
Camada oculta 3: 64 neurônios com função ReLU
Saída: 16 × 4 neurônios
