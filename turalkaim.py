import os
import sys
import math
import random
import numpy as np
import pygame
from pygame import gfxdraw
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam

# -------------------- Global Constants --------------------
BOARD_RADIUS = 6  # side=6 hex => 91 cells total
NN_GRID_SIZE = 2 * BOARD_RADIUS - 1  # 11x11 grid for NN
HEX_SIZE = 40
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)

# Player disc colors
player_colors = {
    0: RED,    # Red
    1: WHITE,  # White
    2: BLACK   # Black
}

BOARD_CELL_FILL = (150, 75, 0)   # Yellow fill
BOARD_BORDER_COLOR = (0, 0, 255)  # Blue border
BG_COLOR = WHITE
TEXT_COLOR = (20, 20, 20)

# Directions for hex adjacency
AXIAL_DIRS = [(1, 0), (1, -1), (0, -1),
              (-1, 0), (-1, 1), (0, 1)]

# -------------------- Board & Game Logic --------------------
def generate_hex_board(radius):
    """Generate all axial coords (q,r) for a hex board of side 'radius'."""
    cells = []
    for q in range(-radius + 1, radius):
        for r in range(-radius + 1, radius):
            if max(abs(q), abs(r), abs(-q - r)) < radius:
                cells.append((q, r))
    return cells

class Board:
    def __init__(self):
        # Valid hex cells
        self.cells = generate_hex_board(BOARD_RADIUS)
        # cell -> -1 if empty, else 0/1/2 for Red/White/Black
        self.board = {cell: -1 for cell in self.cells}

        # "First move" flags so that the first move doesn't actually flip discs
        self.first_move_done = {0: False, 1: False, 2: False}

        self.init_board()

    def init_board(self):
        """
        Place three discs in center:
          F5 → Red   = (0, -1)
          F6 → White = (1, -1)
          G5 → Black = (0, 0)
        """
        try:
            self.board[(0, -1)] = 0  # Red
            self.board[(1, -1)] = 1  # White
            self.board[(0, 0)]  = 2  # Black
        except KeyError:
            print("Error: Could not place initial discs. Check coordinates.")

    def on_board(self, coord):
        return coord in self.board

    def outflanks_in_direction(self, coord, direction, player):
        """
        Standard Othello outflank check in one direction:
          - Must see an opponent disc first,
          - Then zero or more opponent discs,
          - Then eventually a disc of 'player'.
        """
        dq, dr = direction
        q, r = coord
        first_step = (q + dq, r + dr)
        if not self.on_board(first_step):
            return False
        if self.board[first_step] == -1 or self.board[first_step] == player:
            return False

        count_opponent = 0
        current = first_step
        while self.on_board(current) and self.board[current] != -1:
            if self.board[current] == player:
                return count_opponent > 0
            else:
                count_opponent += 1
            cq, cr = current
            current = (cq + dq, cr + dr)
        return False

    def is_valid_move(self, coord, player):
        """
        Each move must be in an outflanking position.
        - That is, for at least one direction, outflanks_in_direction(...) is True.
        The difference is only that for a player's first move, we won't actually flip discs.
        """
        if self.board[coord] != -1:
            return False

        # Each move must outflank in at least one direction
        for d in AXIAL_DIRS:
            if self.outflanks_in_direction(coord, d, player):
                return True
        return False

    def get_valid_moves(self, player):
        return [c for c in self.cells if self.board[c] == -1 and self.is_valid_move(c, player)]

    def apply_move(self, coord, player):
        """
        Place disc. If it's that player's first move, do NOT flip discs.
        If it's not their first move, flip discs in all directions that outflank.
        """
        if not self.is_valid_move(coord, player):
            raise ValueError("apply_move: invalid move.")
        self.board[coord] = player

        if not self.first_move_done[player]:
            # On the first move, do not flip discs, just mark done
            self.first_move_done[player] = True
        else:
            # Subsequent moves: flip if outflanked
            for d in AXIAL_DIRS:
                if self.outflanks_in_direction(coord, d, player):
                    self.flip_direction(coord, d, player)

    def flip_direction(self, coord, direction, player):
        dq, dr = direction
        q, r = coord
        flipping = []
        curr = (q + dq, r + dr)
        while self.on_board(curr) and self.board[curr] != -1:
            if self.board[curr] == player:
                for f in flipping:
                    self.board[f] = player
                return
            else:
                flipping.append(curr)
            cq, cr = curr
            curr = (cq + dq, cr + dr)

    def game_over(self):
        """Game ends if no player has a valid move."""
        for p in [0, 1, 2]:
            if self.get_valid_moves(p):
                return False
        return True

    def get_winner(self):
        """Who has the most discs? Return None if tie."""
        counts = {0:0, 1:0, 2:0}
        for v in self.board.values():
            if v in counts:
                counts[v] += 1
        mx = max(counts.values())
        winners = [p for p,c in counts.items() if c == mx]
        if len(winners) == 1:
            return winners[0]
        return None

    def copy(self):
        nb = Board()
        nb.board = self.board.copy()
        nb.first_move_done = self.first_move_done.copy()
        return nb

# -------------------- NN & MCTS --------------------
def board_to_nn_input(board: Board):
    """Map board to 11x11x3. Unused cells remain 0."""
    tensor = np.zeros((NN_GRID_SIZE, NN_GRID_SIZE, 3))
    offset = BOARD_RADIUS - 1
    for (q,r) in board.cells:
        col = q + offset
        row = r + offset
        if 0 <= col < NN_GRID_SIZE and 0 <= row < NN_GRID_SIZE:
            val = board.board[(q, r)]
            if val in [0,1,2]:
                tensor[row, col, val] = 1
    return tensor

from tensorflow.keras import Model

class NeuralNetwork:
    def __init__(self, input_shape=(NN_GRID_SIZE, NN_GRID_SIZE, 3), num_players=3):
        self.input_shape = input_shape
        self.num_players = num_players
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_players, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
        return model

    def predict(self, board_state):
        board_state = np.expand_dims(board_state, axis=0)
        return self.model.predict(board_state, verbose=0)[0]

    def train(self, x, y, epochs=1):
        self.model.fit(x, y, epochs=epochs, verbose=0)

    def save(self, filename="model.h5"):
        self.model.save(filename)

    def load(self, filename="model.h5"):
        self.model = load_model(filename)

class MCTSNode:
    def __init__(self, board: Board, player, move=None, parent=None):
        self.board = board
        self.player = player
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value_sum = np.zeros(3)
        try:
            self.untried_moves = board.get_valid_moves(player)
        except:
            self.untried_moves = []

class MCTS:
    def __init__(self, neural_net: NeuralNetwork, simulations=100, c_puct=1.0):
        self.nn = neural_net
        self.simulations = simulations
        self.c_puct = c_puct

    def search(self, board: Board, player):
        root = MCTSNode(board.copy(), player)
        if not root.untried_moves:
            return None
        for _ in range(self.simulations):
            node = root
            # Selection
            while not node.untried_moves and node.children:
                node = self.select_child(node)
            # Expansion
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                new_board = node.board.copy()
                try:
                    new_board.apply_move(move, node.player)
                except:
                    node.untried_moves.remove(move)
                    continue
                nxt_p = (node.player + 1) % 3
                child = MCTSNode(new_board, nxt_p, move, node)
                node.children.append(child)
                node.untried_moves.remove(move)
                node = child
            # Evaluation
            val = self.evaluate(node.board)
            self.backpropagate(node, val)

        if not root.children:
            return None
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def select_child(self, node: MCTSNode):
        best_score = -float('inf')
        best_child = None
        for c in node.children:
            Q = c.value_sum[node.player] / (c.visits + 1e-6)
            U = self.c_puct * math.sqrt(node.visits) / (1 + c.visits)
            score = Q + U
            if score > best_score:
                best_score = score
                best_child = c
        return best_child

    def evaluate(self, board: Board):
        if board.game_over():
            w = board.get_winner()
            vec = np.zeros(3)
            if w is not None:
                vec[w] = 1
            return vec
        inp = board_to_nn_input(board)
        return self.nn.predict(inp)

    def backpropagate(self, node: MCTSNode, value):
        while node:
            node.visits += 1
            node.value_sum += value
            node = node.parent

# -------------------- Self-Play Training --------------------
def self_play(neural_net: NeuralNetwork, num_games=10, simulations=50):
    training_examples = []
    for _ in range(num_games):
        board = Board()
        current_player = 0
        game_history = []
        while not board.game_over():
            valid = board.get_valid_moves(current_player)
            if not valid:
                # skip
                current_player = (current_player + 1) % 3
                continue
            mcts = MCTS(neural_net, simulations=simulations)
            move = mcts.search(board, current_player)
            # record board state
            game_history.append((board_to_nn_input(board), current_player))

            if move is None:
                current_player = (current_player + 1) % 3
                continue
            try:
                board.apply_move(move, current_player)
            except Exception as e:
                print("Error applying move in self_play:", e)
                break
            current_player = (current_player + 1) % 3

        winner = board.get_winner()
        result = np.zeros(3)
        if winner is not None:
            result[winner] = 1
        for (st, p) in game_history:
            training_examples.append((st, result))

    if training_examples:
        X = np.array([ex[0] for ex in training_examples])
        Y = np.array([ex[1] for ex in training_examples])
        neural_net.train(X, Y, epochs=1)

# -------------------- Pygame UI & Game Loop --------------------
def axial_to_pixel(q, r, hex_size, origin):
    x = hex_size * 1.5 * q
    y = hex_size * math.sqrt(3) * (r + q / 2)
    return (int(x + origin[0]), int(y + origin[1]))

def polygon_corners(center, size):
    corners = []
    for i in range(6):
        angle = math.pi / 3 * i + math.pi / 6
        cx = center[0] + size * math.cos(angle)
        cy = center[1] + size * math.sin(angle)
        corners.append((int(cx), int(cy)))
    return corners

def draw_text(surface, text, pos, font_size=24, color=TEXT_COLOR):
    font = pygame.font.SysFont("arial", font_size)
    tsurf = font.render(text, True, color)
    surface.blit(tsurf, pos)

def draw_board(screen, board: Board, cell_positions):
    for coord, center in cell_positions.items():
        corners = polygon_corners(center, HEX_SIZE - 2)
        gfxdraw.aapolygon(screen, corners, BOARD_BORDER_COLOR)
        gfxdraw.filled_polygon(screen, corners, BOARD_CELL_FILL)

        val = board.board[coord]
        if val in [0,1,2]:
            disc_color = player_colors[val]
            inner = polygon_corners(center, HEX_SIZE - 12)
            gfxdraw.aapolygon(screen, inner, disc_color)
            gfxdraw.filled_polygon(screen, inner, disc_color)

def precompute_cell_positions(board: Board, origin):
    d = {}
    for c in board.cells:
        d[c] = axial_to_pixel(c[0], c[1], HEX_SIZE, origin)
    return d

def play_game(neural_net: NeuralNetwork, human_player=None):
    """Play a 3-player game. If human_player is None, all 3 are AIs."""
    board = Board()
    current_player = 0
    origin = (WINDOW_WIDTH//2, WINDOW_HEIGHT//2)
    cell_positions = precompute_cell_positions(board, origin)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("3-Player Hex Othello")
    clock = pygame.time.Clock()

    while True:
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        board = Board()
                        current_player = 0
                        cell_positions = precompute_cell_positions(board, origin)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if human_player is not None and current_player == human_player:
                        pos = pygame.mouse.get_pos()
                        clicked = None
                        for coord, center in cell_positions.items():
                            dist = math.hypot(center[0]-pos[0], center[1]-pos[1])
                            if dist < HEX_SIZE:
                                clicked = coord
                                break
                        if clicked and clicked in board.get_valid_moves(current_player):
                            board.apply_move(clicked, current_player)
                            current_player = (current_player + 1) % 3

            # ---------- FIX: if the human has no valid moves, ask them to press enter and skip ----------
            if human_player is not None and current_player == human_player:
                valid_moves = board.get_valid_moves(current_player)
                if not valid_moves:
                    input("You have no valid moves. Press Enter to skip your turn...")
                    current_player = (current_player + 1) % 3
                    # If skipping means all players are out of moves, end now.
                    if board.game_over():
                        winner = board.get_winner()
                        if winner is not None:
                            msg = f"Game Over! Winner: {'Red' if winner==0 else 'White' if winner==1 else 'Black'}"
                        else:
                            msg = "Game Over! It's a tie."
                        print(msg)
                        screen.fill(BG_COLOR)
                        draw_text(screen, msg, (WINDOW_WIDTH//2 - 150, WINDOW_HEIGHT//2 - 20), font_size=32)
                        pygame.display.flip()
                        pygame.time.wait(3000)
                        return

            # If AI turn (or if human had moves, or if we didn't skip)
            if human_player is None or current_player != human_player:
                valid_moves = board.get_valid_moves(current_player)
                if valid_moves:
                    mcts = MCTS(neural_net, simulations=50)
                    move = mcts.search(board, current_player)
                    if move is not None:
                        board.apply_move(move, current_player)
                current_player = (current_player + 1) % 3

            screen.fill(BG_COLOR)
            draw_board(screen, board, cell_positions)
            turn_str = f"Turn: {'Red' if current_player==0 else 'White' if current_player==1 else 'Black'}"
            draw_text(screen, turn_str, (10,10))
            draw_text(screen, "Press R to restart", (10,40))
            if human_player is not None:
                draw_text(screen, "Click a cell to play", (10,70))

            pygame.display.flip()
            clock.tick(5)

            # Check game over for all players.
            if board.game_over():
                winner = board.get_winner()
                if winner is not None:
                    msg = f"Game Over! Winner: {'Red' if winner==0 else 'White' if winner==1 else 'Black'}"
                else:
                    msg = "Game Over! It's a tie."
                print(msg)
                screen.fill(BG_COLOR)
                draw_text(screen, msg, (WINDOW_WIDTH//2 - 150, WINDOW_HEIGHT//2 - 20), font_size=32)
                pygame.display.flip()
                pygame.time.wait(3000)
                return
        except Exception as e:
            print("Error in game loop:", e)
            pygame.quit()
            sys.exit()

# -------------------- Main Menu --------------------
def main():
    """Main loop with training choice, model loading, mode selection, and replay prompt."""
    while True:
        # 1) Ask if we want to train
        try:
            train_choice = input("\nTrain the model? (y/n): ").strip().lower()
        except:
            train_choice = 'n'
        net = NeuralNetwork()

        if train_choice == 'y':
            # ask how many games
            try:
                n_train = int(input("How many self-play games to train on? ").strip())
            except:
                n_train = 0
            for i in range(n_train):
                print(f"Self-play training game {i+1}/{n_train} ...")
                self_play(net, num_games=1, simulations=50)
            # Save
            try:
                net.save("model.h5")
                print("Model saved to model.h5")
            except Exception as e:
                print("Error saving model:", e)
        else:
            # Not training => ask if load saved model
            try:
                load_choice = input("Load saved model? (y/n): ").strip().lower()
            except:
                load_choice = 'n'
            if load_choice=='y' and os.path.exists("model.h5"):
                try:
                    net.load("model.h5")
                    print("Model loaded.")
                except Exception as e:
                    print("Error loading model:", e)
            else:
                print("Using untrained model.")

        # 2) Pick mode
        try:
            mode = input("Pick mode: (1) Watch 3 AIs, (2) Play as Red vs 2 AIs: ").strip()
        except:
            mode = '1'
        if mode=='1':
            play_game(net, human_player=None)
        elif mode=='2':
            play_game(net, human_player=0)
        else:
            print("Invalid mode. Skipping game...")

        # 3) After game ends, ask if we want to relaunch
        try:
            again = input("Demonstrate again? (y/n): ").strip().lower()
        except:
            again = 'n'
        if again!='y':
            print("Exiting.")
            break

if __name__=="__main__":
    main()
