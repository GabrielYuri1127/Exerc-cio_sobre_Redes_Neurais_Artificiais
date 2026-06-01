# Relatório – Resolução de Sudoku 4x4 com Rede Neural Artificial Multicamadas

## Disciplina

Inteligência Artificial

## Professor

Edjard Mota

## Integrantes

- Gabriel Yuri Cavalcante de Castro – 22350996
- Marcele Azevedo de Paula Oliveira – 22353160

## Universidade

Universidade Federal do Amazonas (UFAM)

---

## 1. Introdução

O presente trabalho tem como objetivo propor uma solução computacional baseada em Rede Neural Artificial Multicamadas para resolver o quebra-cabeça Sudoku 4x4, composto por subgrupos 2x2 e valores pertencentes ao conjunto S = {1, 2, 3, 4}.

O Sudoku, apesar de ser conhecido como um jogo lógico, pode ser analisado como um problema clássico de Inteligência Artificial. Sua resolução exige que um conjunto de restrições seja satisfeito simultaneamente. Cada célula deve conter apenas um número, e esse número não pode se repetir na mesma linha, na mesma coluna nem no mesmo subgrupo 2x2.

Dessa forma, o problema não envolve apenas reconhecimento de padrões. Ele também exige raciocínio baseado em regras, restrições e validação lógica. Por esse motivo, a solução desenvolvida combina uma Rede Neural Artificial Multicamadas com um mecanismo simbólico de validação e correção por backtracking.

Essa abordagem híbrida permite explorar tanto o aprendizado estatístico da rede neural quanto a garantia lógica oferecida pela verificação das regras do Sudoku.

---

## 2. Fundamentação Teórica

### 2.1 Sudoku como Problema de Satisfação de Restrições

O Sudoku pode ser definido como um Problema de Satisfação de Restrições, também conhecido como CSP (Constraint Satisfaction Problem). Nesse tipo de problema, há variáveis, domínios e restrições.

No Sudoku 4x4:

- As variáveis são as 16 células do tabuleiro.
- O domínio de cada variável é o conjunto S = {1, 2, 3, 4}.
- As restrições determinam que não pode haver repetição em linhas, colunas e blocos 2x2.

Assim, uma solução só é considerada válida quando todas as restrições são satisfeitas ao mesmo tempo.

### 2.2 Sudoku como Problema de Raciocínio

O Sudoku também pode ser entendido como um problema de raciocínio lógico. Para preencher corretamente uma célula, é necessário analisar quais valores já aparecem na linha, na coluna e no bloco correspondente.

Isso mostra que a escolha de um valor depende do contexto do tabuleiro. Portanto, o Sudoku não é apenas um problema de classificação simples, mas um problema que exige inferência a partir de restrições.

### 2.3 Redes Neurais Artificiais Multicamadas

As Redes Neurais Artificiais Multicamadas, ou MLPs (Multilayer Perceptrons), são modelos compostos por uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída.

Durante o treinamento, a rede ajusta seus pesos internos para reduzir o erro entre a saída prevista e a saída correta. Neste trabalho, a rede foi treinada de forma supervisionada, recebendo como entrada tabuleiros incompletos e como saída esperada seus respectivos tabuleiros completos.

### 2.4 Abordagem Híbrida e IA Neurosimbólica

Uma limitação das redes neurais é que elas aprendem padrões estatísticos, mas não garantem sozinhas o cumprimento de regras lógicas. No caso do Sudoku, isso significa que a RNA pode prever um número repetido em uma linha, coluna ou bloco.

Para lidar com essa limitação, foi utilizada uma abordagem híbrida. A rede neural indica valores prováveis para cada célula, enquanto um validador simbólico verifica se a solução respeita as regras do Sudoku. Caso haja inconsistências, um algoritmo de backtracking guiado pelas probabilidades da rede busca uma solução válida.

Essa combinação aproxima o projeto da ideia de IA neurossimbólica, em que aprendizado de máquina e representação simbólica são usados em conjunto.

---

## 3. Representação do Problema

O tabuleiro utilizado possui dimensão 4x4, totalizando 16 células. As células vazias são representadas pelo valor 0.

Exemplo de tabuleiro incompleto:

```text
1 0 0 4
0 4 1 0
0 1 4 0
4 0 0 1
