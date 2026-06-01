# RNA Multicamadas para Sudoku 4x4

Este projeto resolve o quebra-cabeça Sudoku 4x4 com subgrupos 2x2 usando uma Rede Neural Artificial Multicamadas em Python.

A solução usa uma abordagem híbrida:

1. Uma RNA MLP aprende a prever a solução completa a partir de tabuleiros incompletos.
2. Um validador simbólico verifica as regras do Sudoku.
3. Um backtracking guiado pelas probabilidades da RNA corrige a saída final, preservando as pistas iniciais.

## Regras atendidas

- Cada célula possui apenas um número do conjunto S = {1, 2, 3, 4}.
- Nenhuma linha possui repetição.
- Nenhuma coluna possui repetição.
- Nenhum bloco 2x2 possui repetição.
- Cada linha e coluna da grade 4x4 possui exatamente os números de S.

## Estrutura

```text
sudoku_rna_final/
├── sudoku_rna.py
├── requirements.txt
├── README.md
├── relatorio.md
├── execucao.txt
```

## Como testar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o projeto

```bash
python sudoku_rna.py
```

### 3. Resultado esperado

O programa deve mostrar:

- tabuleiro inicial com zeros;
- saída direta da RNA;
- validação da saída direta;
- solução final corrigida por raciocínio simbólico;
- confirmação de que a solução final é válida.

Exemplo esperado:

```text
Solucao final e valida? True
```

## Observação importante

A RNA sozinha pode gerar saídas inválidas, pois redes neurais aprendem padrões estatísticos, mas não garantem restrições lógicas. Por isso, a solução final combina a RNA com validação simbólica e backtracking. Essa abordagem está alinhada com a ideia de IA neurossimbólica.
