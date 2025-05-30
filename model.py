import math
import random
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.initializers import HeNormal

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available - using standard Python implementation")

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

try:
    CPU_COUNT = max(2, multiprocessing.cpu_count() // 2)
    CPU_COUNT = min(CPU_COUNT, 4)
    THREAD_POOL = ThreadPoolExecutor(max_workers=CPU_COUNT)
except:
    CPU_COUNT = 2
    THREAD_POOL = None
    print("Thread pool setup failed - using sequential processing")

BOARD_RADIUS = 6  
NN_GRID_SIZE = 2 * BOARD_RADIUS - 1  

def board_to_nn_input(board):
    
    #Convert the board state into an 11x11x3 tensor.
    
    tensor = np.zeros((NN_GRID_SIZE, NN_GRID_SIZE, 3))
    offset = BOARD_RADIUS - 1
    for (q, r) in board.cells:
        col = q + offset
        row = r + offset
        if 0 <= col < NN_GRID_SIZE and 0 <= row < NN_GRID_SIZE:
            val = board.board[(q, r)]
            if val in [0, 1, 2]:
                tensor[row, col, val] = 1
    return tensor

class NeuralNetwork:
    def __init__(self, input_shape=(NN_GRID_SIZE, NN_GRID_SIZE, 3), num_players=3, learning_rate=0.0001):
        self.input_shape = input_shape
        self.num_players = num_players
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target()

    def build_model(self):
        reg_strength = 0.003  
        
        # Input layer
        input_layer = Input(shape=self.input_shape)
        
        # Shared layers
        x = Conv2D(64, kernel_size=3, activation='relu', padding='same', 
                   kernel_regularizer=l2(reg_strength),
                   kernel_initializer=HeNormal())(input_layer)
        x = BatchNormalization()(x)
        
        x = Conv2D(64, kernel_size=3, activation='relu', padding='same',
                   kernel_regularizer=l2(reg_strength),
                   kernel_initializer=HeNormal())(x)
        x = BatchNormalization()(x)
        
        x = Flatten()(x)
        
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        
        x = Dense(128, activation='relu',
                 kernel_regularizer=l2(reg_strength),
                 kernel_initializer=HeNormal())(x)
        x = BatchNormalization()(x)
        
        # Policy head 
        policy_head = Dense(self.num_players, activation='softmax', 
                          kernel_regularizer=l2(reg_strength),
                          kernel_initializer=HeNormal(),
                          name='policy')(x)
        
        # Value head 
        value_stream = Dense(128, activation='relu', kernel_regularizer=l2(reg_strength))(x)
        value_stream = BatchNormalization()(value_stream)
        value_stream = Dropout(0.2)(value_stream)
        value_head = Dense(self.num_players, activation='softmax', name='value')(value_stream)
        
        model = Model(inputs=input_layer, outputs=[policy_head, value_head])
        
        model.compile(
            loss={
                'policy': CategoricalCrossentropy(label_smoothing=0.05),
                'value': CategoricalCrossentropy(label_smoothing=0.1)
            },
            optimizer=Adam(learning_rate=self.learning_rate, clipnorm=0.5),
            metrics={
                'policy': ['accuracy'],
                'value': ['accuracy']
            }
        )
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def predict(self, board_state, batch=False):
        if not batch:
            board_state = np.expand_dims(board_state, axis=0)
        policy, value = self.model.predict(board_state, verbose=0, batch_size=len(board_state))
        if not batch:
            return policy[0], value[0]
        return policy, value

    def predict_target(self, board_state, batch=False):
        if not batch:
            board_state = np.expand_dims(board_state, axis=0)
        policy, value = self.target_model.predict(board_state, verbose=0, batch_size=len(board_state))
        if not batch:
            return policy[0], value[0]
        return policy, value

    def train(self, x, y_policy, y_value, epochs=1, validation_split=0.1, batch_size=64):
        return self.model.fit(
            x, 
            {'policy': y_policy, 'value': y_value}, 
            epochs=epochs, 
            batch_size=batch_size,
            verbose=0, 
            validation_split=validation_split
        )

    def save(self, filename="model.h5"):
        self.model.save(filename)

    def load(self, filename="model.h5"):
        from tensorflow.keras.models import load_model as keras_load_model
        self.model = keras_load_model(filename)
        self.model.compile(
            loss={
                'policy': CategoricalCrossentropy(label_smoothing=0.1),
                'value': CategoricalCrossentropy(label_smoothing=0.15)
            },
            optimizer=Adam(learning_rate=self.learning_rate, clipnorm=0.5),
            metrics={
                'policy': ['accuracy'],
                'value': ['accuracy']
            }
        )
        self.target_model = keras_load_model(filename)

class MCTSNode:
    #node in the Monte Carlo Tree Search.
    
    def __init__(self, board, player, move=None, parent=None):
        self.board = board  
        self.player = player
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value_sum = np.zeros(3)
        try:
            self.untried_moves = board.get_valid_moves(player)
        except Exception:
            self.untried_moves = []

class MCTS:
    def __init__(self, neural_net, simulations=100, c_puct=1.0, temperature=1.0):
        self.nn = neural_net
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.evaluation_cache = {}

    def search(self, board, player, return_probs=False):
        """
        MCTS search and return the best move.
        
        """
        valid_moves = board.get_valid_moves(player)
        if not valid_moves:
            return None if not return_probs else (None, {})
        
        root = MCTSNode(board.copy(), player)
        if not root.untried_moves:
            return None if not return_probs else (None, {})

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
                except Exception:
                    node.untried_moves.remove(move)
                    continue
                nxt_p = (node.player + 1) % 3
                child = MCTSNode(new_board, nxt_p, move, node)
                node.children.append(child)
                node.untried_moves.remove(move)
                node = child
            # target network.
            value = self.evaluate(node.board)
            self.backpropagate(node, value)

        if not root.children:
            return None if not return_probs else (None, {})
            
        if return_probs:
            move_counts = {child.move: child.visits for child in root.children}
            
            # temperature scaling
            total_visits = sum(move_counts.values())
            if total_visits > 0:
                
                if self.temperature == 0:
                    best_move = max(move_counts.items(), key=lambda x: x[1])[0]
                    move_probs = {move: 1.0 if move == best_move else 0.0 for move in move_counts.keys()}
                else:
                    move_probs = {}
                    for move, count in move_counts.items():
                        if self.temperature < 0.01:
                            move_probs[move] = 0.0
                        else:
                            move_probs[move] = (count ** (1.0 / self.temperature)) / total_visits
                    
                    prob_sum = sum(move_probs.values())
                    if prob_sum > 0:
                        move_probs = {m: p / prob_sum for m, p in move_probs.items()}
            else:
                move_probs = {move: 1.0 / len(move_counts) for move in move_counts.keys()}
                
            if self.temperature == 0:
                best_child = max(root.children, key=lambda c: c.visits)
                selected_move = best_child.move
            else:
                # temperature scaling to visit counts
                visits = np.array([child.visits for child in root.children])
                
                if self.temperature < 0.01:
                    best_idx = np.argmax(visits)
                    selected_move = root.children[best_idx].move
                else:
                    # temperature scaling
                    scaled_visits = visits ** (1.0 / self.temperature)
                    visit_probs = scaled_visits / scaled_visits.sum()
                    
                    chosen_idx = np.random.choice(len(root.children), p=visit_probs)
                    selected_move = root.children[chosen_idx].move
                    
            return selected_move, move_probs
        else:
            if self.temperature == 0:
                best_child = max(root.children, key=lambda c: c.visits)
                return best_child.move
            else:
                visits = np.array([child.visits for child in root.children])
                
                if self.temperature < 0.01:
                    best_idx = np.argmax(visits)
                    return root.children[best_idx].move
                    
                scaled_visits = visits ** (1.0 / self.temperature)
                visit_probs = scaled_visits / scaled_visits.sum()
                
                chosen_idx = np.random.choice(len(root.children), p=visit_probs)
                return root.children[chosen_idx].move

    def select_child(self, node):
        best_score = -float('inf')
        best_child = None
        for child in node.children:
            Q = child.value_sum[node.player] / (child.visits + 1e-6)
            U = self.c_puct * math.sqrt(node.visits) / (1 + child.visits)
            score = Q + U
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    @lru_cache(maxsize=1024)
    def _evaluate_cached(self, board_str):
        ##Cached evaluation function 
        
        return self.nn.predict_target(self.board_state_cache[board_str])
    
    def evaluate(self, board):
        #evaluate a board position with caching
        if board.game_over():
            winner = board.get_winner()
            value_vector = np.zeros(3)
            if winner is not None:
                for p in range(3):
                    distance = (winner - p) % 3  # distance from winner (0=winner, 1=next, 2=furthest)
                    value_vector[p] = max(0, 1.0 - distance * 0.5)  # 1.0, 0.5, 0.0 for winner, next, furthest
            return value_vector
            
        board_str = str(sorted(board.board.items()))
        
        if board_str in self.evaluation_cache:
            return self.evaluation_cache[board_str]
        
        nn_input = board_to_nn_input(board)
        
        if not hasattr(self, 'board_state_cache'):
            self.board_state_cache = {}
        self.board_state_cache[board_str] = nn_input
        
        _, value = self._evaluate_cached(board_str)
        
        self.evaluation_cache[board_str] = value
        return value
        
    def evaluate_batch(self, boards):
        #multiple board positions at once
        terminal_values = []
        non_terminal_boards = []
        non_terminal_indices = []
        
        for i, board in enumerate(boards):
            if board.game_over():
                winner = board.get_winner()
                value_vector = np.zeros(3)
                if winner is not None:
                    for p in range(3):
                        distance = (winner - p) % 3  
                        value_vector[p] = max(0, 1.0 - distance * 0.5) 
                terminal_values.append((i, value_vector))
            else:
                non_terminal_boards.append(board)
                non_terminal_indices.append(i)
        
        
        if non_terminal_boards:
            inputs = np.array([board_to_nn_input(b) for b in non_terminal_boards])
            _, values = self.nn.predict_target(inputs, batch=True)
            
            for idx, board in enumerate(non_terminal_boards):
                board_str = str(sorted(board.board.items()))
                self.evaluation_cache[board_str] = values[idx]
        
        # combine results
        all_values = [None] * len(boards)
        for i, val in terminal_values:
            all_values[i] = val
            
        for batch_idx, orig_idx in enumerate(non_terminal_indices):
            all_values[orig_idx] = values[batch_idx]
            
        return all_values

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value_sum += value
            node = node.parent
