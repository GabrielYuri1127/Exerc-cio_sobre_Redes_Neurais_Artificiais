"""
Exercicio: RNA multicamadas para Sudoku 4x4.

A solucao combina:
1) Geracao de dados validos de Sudoku 4x4.
2) Rede neural MLP em PyTorch para prever a solucao completa.
3) Validador simbolico das regras do Sudoku.
4) Busca/backtracking guiada pelas probabilidades da RNA para garantir tabuleiro final valido.
"""

import itertools
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

N = 4
SUB = 2
VALORES = [1, 2, 3, 4]
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)


BASE_BOARD = np.array([
    [1, 2, 3, 4],
    [3, 4, 1, 2],
    [2, 1, 4, 3],
    [4, 3, 2, 1],
], dtype=int)


def is_valid_board(board: np.ndarray) -> bool:
    """Verifica se um tabuleiro 4x4 atende todas as restricoes do Sudoku."""
    board = np.array(board)
    alvo = set(VALORES)

    if board.shape != (N, N):
        return False

    for i in range(N):
        if set(board[i, :]) != alvo:
            return False
        if set(board[:, i]) != alvo:
            return False

    for r in range(0, N, SUB):
        for c in range(0, N, SUB):
            bloco = board[r:r + SUB, c:c + SUB].reshape(-1)
            if set(bloco) != alvo:
                return False

    return True


def generate_all_solution_boards() -> List[np.ndarray]:
    """Gera todos os tabuleiros completos validos 4x4 por backtracking."""
    boards: List[np.ndarray] = []
    board = np.zeros((N, N), dtype=int)

    def valid_place(r: int, c: int, v: int) -> bool:
        if v in board[r, :]:
            return False
        if v in board[:, c]:
            return False
        br, bc = (r // SUB) * SUB, (c // SUB) * SUB
        if v in board[br:br+SUB, bc:bc+SUB]:
            return False
        return True

    def backtrack(pos: int):
        if pos == N * N:
            boards.append(board.copy())
            return
        r, c = divmod(pos, N)
        for v in VALORES:
            if valid_place(r, c, v):
                board[r, c] = v
                backtrack(pos + 1)
                board[r, c] = 0

    backtrack(0)
    return boards


def mask_board(solution: np.ndarray, min_clues: int = 4, max_clues: int = 10) -> np.ndarray:
    """Remove celulas aleatorias do tabuleiro completo. Zero representa celula vazia."""
    puzzle = solution.copy()
    clues = random.randint(min_clues, max_clues)
    positions = list(range(N * N))
    random.shuffle(positions)
    remove_count = N * N - clues
    for idx in positions[:remove_count]:
        r, c = divmod(idx, N)
        puzzle[r, c] = 0
    return puzzle


def one_hot_input(puzzle: np.ndarray) -> np.ndarray:
    """Codifica cada celula em one-hot com 5 possibilidades: vazio, 1, 2, 3, 4."""
    x = np.zeros((N * N, N + 1), dtype=np.float32)
    flat = puzzle.reshape(-1)
    for i, value in enumerate(flat):
        x[i, int(value)] = 1.0
    return x.reshape(-1)


def target_output(solution: np.ndarray) -> np.ndarray:
    """Rotulos de 0 a 3 representando os valores 1 a 4 em cada celula."""
    return solution.reshape(-1).astype(np.int64) - 1


def make_dataset(num_samples: int = 6000) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    solutions = generate_all_solution_boards()
    X, y = [], []
    for _ in range(num_samples):
        sol = random.choice(solutions)
        puzzle = mask_board(sol)
        X.append(one_hot_input(puzzle))
        y.append(target_output(sol))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), solutions


class SudokuMLP(nn.Module):
    """Rede neural artificial multicamadas para prever valores das 16 celulas."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16 * 4)
        )

    def forward(self, x):
        return self.net(x).view(-1, 16, 4)


@dataclass
class TrainResult:
    model: SudokuMLP
    train_losses: List[float]
    test_accuracies: List[float]


def train_model(epochs: int = 80, num_samples: int = 8000) -> Tuple[TrainResult, Tuple[np.ndarray, np.ndarray], List[np.ndarray]]:
    X, y, solutions = make_dataset(num_samples)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    model = SudokuMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses, accs = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits.reshape(-1, 4), yb.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test))
            pred = logits.argmax(dim=-1)
            acc = (pred == torch.tensor(y_test)).float().mean().item()

        losses.append(total_loss / len(train_loader))
        accs.append(acc)
        if (epoch + 1) % 10 == 0:
            print(f"Epoca {epoch+1:03d} | loss={losses[-1]:.4f} | acc_teste={acc:.4f}")

    return TrainResult(model, losses, accs), (X_test, y_test), solutions


def candidates_for_cell(board: np.ndarray, r: int, c: int) -> List[int]:
    if board[r, c] != 0:
        return [int(board[r, c])]
    used = set(board[r, :]) | set(board[:, c]) | set(board[(r//SUB)*SUB:(r//SUB)*SUB+SUB, (c//SUB)*SUB:(c//SUB)*SUB+SUB].reshape(-1))
    return [v for v in VALORES if v not in used]


def solve_with_backtracking_guided_by_rna(puzzle: np.ndarray, probs: np.ndarray) -> Optional[np.ndarray]:
    """Completa o Sudoku usando restricoes simbolicas e a ordem sugerida pela RNA."""
    board = puzzle.copy()

    def backtrack() -> bool:
        empty_cells = [(r, c) for r in range(N) for c in range(N) if board[r, c] == 0]
        if not empty_cells:
            return is_valid_board(board)

        # MRV: escolhe a celula com menos candidatos; desempate pela confianca da RNA.
        best = None
        best_cands = None
        for r, c in empty_cells:
            cands = candidates_for_cell(board, r, c)
            if not cands:
                return False
            score = max(float(probs[r * N + c, v - 1]) for v in cands)
            key = (len(cands), -score)
            if best is None or key < best:
                best = key
                best_cands = (r, c, cands)

        r, c, cands = best_cands
        cands = sorted(cands, key=lambda v: probs[r * N + c, v - 1], reverse=True)
        for v in cands:
            board[r, c] = v
            if backtrack():
                return True
            board[r, c] = 0
        return False

    if backtrack():
        return board
    return None


def predict_solution(model: SudokuMLP, puzzle: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    x = torch.tensor(one_hot_input(puzzle)).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=-1).numpy()
    raw = probs.argmax(axis=-1).reshape(N, N) + 1

    # Preserva as pistas iniciais.
    raw = raw.astype(int)
    raw[puzzle != 0] = puzzle[puzzle != 0]

    repaired = solve_with_backtracking_guided_by_rna(puzzle, probs)
    return raw, probs, repaired


def print_board(title: str, board: np.ndarray):
    print(title)
    for row in board:
        print(" ".join(str(int(x)) for x in row))
    print()


def plot_training(losses: List[float], accs: List[float]):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Loss de treino")
    plt.plot(accs, label="Acuracia no teste")
    plt.xlabel("Epoca")
    plt.ylabel("Valor")
    plt.title("Treinamento da RNA para Sudoku 4x4")
    plt.legend()
    plt.tight_layout()
    plt.savefig("imagens/curva_treinamento.png", dpi=150)


def main():
    print("Gerando dados e treinando RNA multicamadas...")
    result, _, solutions = train_model(epochs=60, num_samples=4000)
    torch.save(result.model.state_dict(), "modelo_sudoku4x4.pt")
    plot_training(result.train_losses, result.test_accuracies)

    solution = random.choice(solutions)
    puzzle = mask_board(solution, min_clues=5, max_clues=7)
    raw, probs, repaired = predict_solution(result.model, puzzle)

    print_board("Tabuleiro inicial:", puzzle)
    print_board("Saida direta da RNA:", raw)
    print("Saida direta e valida?", is_valid_board(raw))

    if repaired is not None:
        print_board("Solucao final apos validacao simbolica:", repaired)
        print("Solucao final e valida?", is_valid_board(repaired))
    else:
        print("Nao foi possivel reparar a solucao.")

    print("Arquivo de imagem gerado: imagens/curva_treinamento.png")
    print("Modelo salvo: modelo_sudoku4x4.pt")


if __name__ == "__main__":
    main()
