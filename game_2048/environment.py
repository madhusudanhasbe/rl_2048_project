import numpy as np
import random
from .utils import print_board, render_board_as_image

class Game2048Env:
    def __init__(self, size=4):
        self.size = size
        self.board = None
        self.score = 0
        self.done = False
        self.action_space = [0, 1, 2, 3] # 0: Up, 1: Down, 2: Left, 3: Right

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.done = False
        self._add_new_tile()
        self._add_new_tile()
        return self.get_state()

    def _get_empty_cells(self):
        return [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]

    def _add_new_tile(self):
        empty = self._get_empty_cells()
        if empty:
            r, c = random.choice(empty)
            self.board[r, c] = 2 if random.random() < 0.9 else 4

    def get_state(self):
        return np.copy(self.board)

    def _compress(self, row):
        new_row = [i for i in row if i != 0]
        new_row += [0] * (self.size - len(new_row))
        return new_row

    def _merge(self, row):
        row = self._compress(row)
        score_gain = 0
        for i in range(self.size - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                score_gain += row[i]
                row[i + 1] = 0
        return self._compress(row), score_gain

    def _move_left(self):
        new_board = np.zeros_like(self.board)
        score = 0
        for r in range(self.size):
            new_board[r, :], s = self._merge(self.board[r, :])
            score += s
        return new_board, score

    def _move_right(self):
        reversed_board = np.fliplr(self.board)
        temp_board, score = self._move_left()
        return np.fliplr(temp_board), score

    def _move_up(self):
        self.board = self.board.T
        temp_board, score = self._move_left()
        return temp_board.T, score

    def _move_down(self):
        self.board = self.board.T
        temp_board, score = self._move_right()
        return temp_board.T, score

    def _is_game_over(self):
        if self._get_empty_cells():
            return False
        for r in range(self.size):
            for c in range(self.size - 1):
                if self.board[r, c] == self.board[r, c + 1] or self.board[c, r] == self.board[c + 1, r]:
                    return False
        return True

    def get_legal_actions(self):
        legal = []
        for action in self.action_space:
            temp_board, _ = self._perform_move(action)
            if not np.array_equal(self.board, temp_board):
                legal.append(action)
        return legal

    def _perform_move(self, action):
        if action == 0: return self._move_up()
        elif action == 1: return self._move_down()
        elif action == 2: return self._move_left()
        elif action == 3: return self._move_right()
        else: raise ValueError("Invalid action")

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}

        original_board = self.board.copy()
        new_board, reward = self._perform_move(action)
        board_changed = not np.array_equal(original_board, new_board)

        if board_changed:
            self.board = new_board
            self._add_new_tile()
            self.score += reward

        self.done = self._is_game_over()
        return self.get_state(), float(reward), self.done, {'score': self.score, 'board_changed': board_changed}

    def render(self, mode='human'):
        if mode == 'human':
            print_board(self.board)
            print(f"Score: {self.score}\n")
        elif mode == 'rgb_array':
            return render_board_as_image(self.board)
        else:
            raise ValueError("Unsupported render mode")