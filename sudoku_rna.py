import argparse
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def symbols(size):
    return list(range(1, size + 1))


def block_size(size):
    root = int(math.sqrt(size))
    if root * root != size:
        raise ValueError("O tamanho deve ter raiz quadrada inteira: 4, 9 ou 16.")
    return root


def generate_base_board(size):
    b = block_size(size)
    board = np.zeros((size, size), dtype=int)

    for r in range(size):
        for c in range(size):
            board[r, c] = ((r * b + r // b + c) % size) + 1

    return board


def shuffle_board(board):
    size = board.shape[0]
    b = block_size(size)

    new_board = board.copy()

    rows = []
    row_bands = list(range(b))
    random.shuffle(row_bands)

    for band in row_bands:
        band_rows = list(range(band * b, band * b + b))
        random.shuffle(band_rows)
        rows.extend(band_rows)

    cols = []
    col_stacks = list(range(b))
    random.shuffle(col_stacks)

    for stack in col_stacks:
        stack_cols = list(range(stack * b, stack * b + b))
        random.shuffle(stack_cols)
        cols.extend(stack_cols)

    new_board = new_board[rows, :]
    new_board = new_board[:, cols]

    vals = symbols(size)
    shuffled = vals.copy()
    random.shuffle(shuffled)
    mapping = {old: new for old, new in zip(vals, shuffled)}

    for old, new in mapping.items():
        new_board[board == old] = new

    return new_board


def generate_complete_board(size):
    base = generate_base_board(size)
    return shuffle_board(base)


def remove_cells(board, removed_ratio=0.45):
    puzzle = board.copy()
    size = board.shape[0]
    total = size * size
    remove_count = int(total * removed_ratio)

    positions = list(range(total))
    random.shuffle(positions)

    for pos in positions[:remove_count]:
        r, c = divmod(pos, size)
        puzzle[r, c] = 0

    return puzzle


def is_valid_board(board):
    board = np.array(board)
    size = board.shape[0]
    b = block_size(size)
    target = set(symbols(size))

    if board.shape != (size, size):
        return False

    for r in range(size):
        if set(board[r, :]) != target:
            return False

    for c in range(size):
        if set(board[:, c]) != target:
            return False

    for br in range(0, size, b):
        for bc in range(0, size, b):
            block = board[br:br + b, bc:bc + b].flatten()
            if set(block) != target:
                return False

    return True


def one_hot_board(board):
    board = np.array(board)
    size = board.shape[0]
    flat = board.flatten()

    encoded = np.zeros((size * size, size + 1), dtype=np.float32)

    for i, value in enumerate(flat):
        encoded[i, int(value)] = 1.0

    return encoded.flatten()


def generate_dataset(size, samples, removed_ratio):
    x_data = []
    y_data = []

    for _ in range(samples):
        solution = generate_complete_board(size)
        puzzle = remove_cells(solution, removed_ratio)

        x_data.append(one_hot_board(puzzle))
        y_data.append(solution.flatten() - 1)

    x = torch.tensor(np.array(x_data), dtype=torch.float32)
    y = torch.tensor(np.array(y_data), dtype=torch.long)

    return x, y


class SudokuMLP(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.size = size
        input_dim = size * size * (size + 1)
        output_dim = size * size * size

        hidden1 = max(128, input_dim)
        hidden2 = max(128, input_dim // 2)
        hidden3 = max(64, input_dim // 4)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, output_dim)
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.size * self.size, self.size)


def train_model(model, x_train, y_train, x_test, y_test, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        logits = model(x_train)

        loss = criterion(
            logits.reshape(-1, model.size),
            y_train.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            test_logits = model(x_test)
            predictions = torch.argmax(test_logits, dim=2)
            accuracy = (predictions == y_test).float().mean().item()

        print(
            f"Época {epoch + 1:03d}/{epochs} | "
            f"Loss: {loss.item():.4f} | "
            f"Acurácia teste: {accuracy:.4f}"
        )


def predict_board(model, puzzle):
    size = model.size

    model.eval()

    x = torch.tensor(
        one_hot_board(puzzle),
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=2)[0].numpy()
        pred = np.argmax(probs, axis=1) + 1

    predicted_board = pred.reshape(size, size)

    puzzle = np.array(puzzle)

    for r in range(size):
        for c in range(size):
            if puzzle[r, c] != 0:
                predicted_board[r, c] = puzzle[r, c]

    return predicted_board, probs


def is_safe(board, row, col, value):
    size = board.shape[0]
    b = block_size(size)

    if value in board[row, :]:
        return False

    if value in board[:, col]:
        return False

    start_row = (row // b) * b
    start_col = (col // b) * b

    if value in board[start_row:start_row + b, start_col:start_col + b]:
        return False

    return True


def backtracking_guided(puzzle, probs, max_empty_for_backtracking=60):
    board = np.array(puzzle).copy()
    size = board.shape[0]

    empty_cells = []

    for r in range(size):
        for c in range(size):
            if board[r, c] == 0:
                idx = r * size + c
                confidence = np.max(probs[idx])
                empty_cells.append((confidence, r, c))

    if len(empty_cells) > max_empty_for_backtracking:
        print(
            f"\nAviso: {len(empty_cells)} células vazias. "
            f"Backtracking pode ficar pesado. "
            f"Limite atual: {max_empty_for_backtracking}."
        )
        return None

    empty_cells.sort(reverse=True)

    def solve():
        selected = None

        for _, r, c in empty_cells:
            if board[r, c] == 0:
                selected = (r, c)
                break

        if selected is None:
            return is_valid_board(board)

        r, c = selected
        idx = r * size + c

        values = list(range(1, size + 1))
        values.sort(key=lambda v: probs[idx][v - 1], reverse=True)

        for value in values:
            if is_safe(board, r, c, value):
                board[r, c] = value

                if solve():
                    return True

                board[r, c] = 0

        return False

    if solve():
        return board

    return None


def solve_with_neural_and_symbolic(model, puzzle, max_empty_for_backtracking):
    direct_prediction, probs = predict_board(model, puzzle)

    if is_valid_board(direct_prediction):
        return direct_prediction, direct_prediction, True

    corrected = backtracking_guided(
        puzzle,
        probs,
        max_empty_for_backtracking=max_empty_for_backtracking
    )

    if corrected is None:
        return direct_prediction, direct_prediction, False

    return direct_prediction, corrected, is_valid_board(corrected)


def print_board(title, board):
    print(f"\n{title}")
    print("-" * 40)

    size = board.shape[0]

    for r in range(size):
        print(" ".join(f"{int(x):2d}" for x in board[r]))


def main():
    parser = argparse.ArgumentParser(
        description="RNA Multicamadas para Sudoku 4x4, 9x9 e 16x16."
    )

    parser.add_argument("--size", type=int, default=4, choices=[4, 9, 16])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--removed-ratio", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-empty", type=int, default=60)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    size = args.size

    print(f"Sudoku {size}x{size}")
    print(f"Blocos {block_size(size)}x{block_size(size)}")
    print("Gerando dados...")

    x, y = generate_dataset(
        size=size,
        samples=args.samples,
        removed_ratio=args.removed_ratio
    )

    split = int(0.8 * args.samples)

    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Amostras de treino: {len(x_train)}")
    print(f"Amostras de teste: {len(x_test)}")

    model = SudokuMLP(size)

    print("\nTreinando a RNA...")

    train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=args.epochs,
        lr=args.lr
    )

    model_name = f"modelo_sudoku{size}x{size}.pt"
    torch.save(model.state_dict(), model_name)

    print(f"\nModelo salvo em: {model_name}")

    solution = generate_complete_board(size)
    puzzle = remove_cells(solution, removed_ratio=args.removed_ratio)

    direct_prediction, final_solution, valid = solve_with_neural_and_symbolic(
        model,
        puzzle,
        max_empty_for_backtracking=args.max_empty
    )

    print_board("Tabuleiro inicial", puzzle)
    print_board("Solução correta original", solution)
    print_board("Saída direta da RNA", direct_prediction)

    print(f"\nSaída direta da RNA é válida? {is_valid_board(direct_prediction)}")

    print_board("Solução final", final_solution)

    print(f"\nSolução final é válida? {valid}")


if __name__ == "__main__":
    main()
