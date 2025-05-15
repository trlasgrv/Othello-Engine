# model.py

import math
import random
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.initializers import HeNormal
# For performance optimization
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available - using standard Python implementation")
# For CPU performance optimization
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Set up thread pool for parallel operations
try:
    # Use half of available CPU cores
    CPU_COUNT = max(2, multiprocessing.cpu_count() // 2)
    # But not more than 4 for typical applications
    CPU_COUNT = min(CPU_COUNT, 4)
    THREAD_POOL = ThreadPoolExecutor(max_workers=CPU_COUNT)
except:
    CPU_COUNT = 2
    THREAD_POOL = None
    print("Thread pool setup failed - using sequential processing")

# Global constants for the neural network input
BOARD_RADIUS = 6  # side=6 hex => 91 cells total
NN_GRID_SIZE = 2 * BOARD_RADIUS - 1  # 11x11 grid for NN input

def board_to_nn_input(board):
    """
    Convert the board state into an 11x11x3 tensor.
    Assumes:
      - board.cells: list of valid (q, r) coordinates.
      - board.board: dict mapping each cell to -1 (empty) or player index (0, 1, 2).
    """
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
    """
    Neural Network model for board evaluation.
    Uses two convolutional layers (with Batch Normalization) followed by separate policy and value heads.
    The policy head predicts move probabilities while the value head estimates win probabilities.
    
    Includes a target network that is used for MCTS evaluation. The target network is updated only when update_target() is called.
    """
    def __init__(self, input_shape=(NN_GRID_SIZE, NN_GRID_SIZE, 3), num_players=3, learning_rate=0.0001):
        self.input_shape = input_shape
        self.num_players = num_players
        self.learning_rate = learning_rate
        self.model = self.build_model()
        # Create a separate target model with the same architecture.
        self.target_model = self.build_model()
        self.update_target()

    def build_model(self):
        # Higher regularization to prevent overfitting
        reg_strength = 0.003  # Reduced from 0.005 to help with faster progress
        
        # Input layer
        input_layer = Input(shape=self.input_shape)
        
        # Shared layers
        x = Conv2D(64, kernel_size=3, activation='relu', padding='same',  # Increased from 32 to 64
                   kernel_regularizer=l2(reg_strength),
                   kernel_initializer=HeNormal())(input_layer)
        x = BatchNormalization()(x)
        
        x = Conv2D(64, kernel_size=3, activation='relu', padding='same',
                   kernel_regularizer=l2(reg_strength),
                   kernel_initializer=HeNormal())(x)
        x = BatchNormalization()(x)
        
        x = Flatten()(x)
        
        # Add dropout for regularization
        x = Dropout(0.2)(x)  # Reduced from 0.3 for faster progress
        x = BatchNormalization()(x)
        
        x = Dense(128, activation='relu',   # Increased from 64 to 128
                 kernel_regularizer=l2(reg_strength),
                 kernel_initializer=HeNormal())(x)
        x = BatchNormalization()(x)
        
        # Policy head - predicts move probabilities
        policy_head = Dense(self.num_players, activation='softmax', 
                          kernel_regularizer=l2(reg_strength),
                          kernel_initializer=HeNormal(),
                          name='policy')(x)
        
        # Value head - estimates win probabilities (improved)
        value_stream = Dense(128, activation='relu', kernel_regularizer=l2(reg_strength))(x)
        value_stream = BatchNormalization()(value_stream)
        value_stream = Dropout(0.2)(value_stream)  # Less dropout for faster progress
        value_head = Dense(self.num_players, activation='softmax', name='value')(value_stream)
        
        # Create model with two outputs
        model = Model(inputs=input_layer, outputs=[policy_head, value_head])
        
        # Compile with two loss functions
        model.compile(
            loss={
                'policy': CategoricalCrossentropy(label_smoothing=0.05),  # Reduced from 0.1
                'value': CategoricalCrossentropy(label_smoothing=0.1)     # Reduced from 0.15
            },
            optimizer=Adam(learning_rate=self.learning_rate, clipnorm=0.5),
            metrics={
                'policy': ['accuracy'],
                'value': ['accuracy']
            }
        )
        return model

    def update_target(self):
        """Copy weights from the main network to the target network."""
        self.target_model.set_weights(self.model.get_weights())

    def predict(self, board_state, batch=False):
        """
        Predict policy and value for a board state.
        If batch=True, assumes board_state is already a batch of states.
        Otherwise, adds a batch dimension.
        """
        if not batch:
            board_state = np.expand_dims(board_state, axis=0)
        policy, value = self.model.predict(board_state, verbose=0, batch_size=len(board_state))
        if not batch:
            return policy[0], value[0]
        return policy, value

    def predict_target(self, board_state, batch=False):
        """
        Use the target network for prediction.
        If batch=True, assumes board_state is already a batch of states.
        Otherwise, adds a batch dimension.
        """
        if not batch:
            board_state = np.expand_dims(board_state, axis=0)
        policy, value = self.target_model.predict(board_state, verbose=0, batch_size=len(board_state))
        if not batch:
            return policy[0], value[0]
        return policy, value

    def train(self, x, y_policy, y_value, epochs=1, validation_split=0.1, batch_size=64):  # Smaller batch size (64)
        # Use a validation split and return the training history.
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
        # Recompile the model to initialize a fresh optimizer
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
        # Also load into target model but don't need to compile it
        self.target_model = keras_load_model(filename)

class MCTSNode:
    """
    A node in the Monte Carlo Tree Search.
    """
    def __init__(self, board, player, move=None, parent=None):
        self.board = board  # Assumes board is an instance of your Board class.
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
    """
    Monte Carlo Tree Search (MCTS) implementation that uses a neural network for board evaluation.
    Uses temperature-based exploration to control the tradeoff between exploration and exploitation.
    Can return visit counts distribution for policy distillation.
    """
    def __init__(self, neural_net, simulations=100, c_puct=1.0, temperature=1.0):
        self.nn = neural_net
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        # Add prediction cache to avoid redundant calculations
        self.evaluation_cache = {}

    def search(self, board, player, return_probs=False):
        """
        Run MCTS search and return the best move.
        
        Parameters:
        -----------
        board : Board
            The current game board
        player : int
            The current player (0, 1, or 2)
        return_probs : bool
            If True, return (move, probabilities) where probabilities is a
            dictionary mapping moves to their visit probabilities.
            
        Returns:
        --------
        move : tuple or None
            The selected move, or None if no valid moves
        probs : dict, optional
            Only returned if return_probs=True. Dictionary mapping moves to visit probabilities.
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
            # Evaluation using the target network.
            value = self.evaluate(node.board)
            self.backpropagate(node, value)

        if not root.children:
            return None if not return_probs else (None, {})
            
        # If we're generating training data, we want the full probability distribution over moves
        if return_probs:
            # Create a mapping of moves to visit counts
            move_counts = {child.move: child.visits for child in root.children}
            
            # Apply temperature scaling
            total_visits = sum(move_counts.values())
            if total_visits > 0:
                # Convert to probabilities
                if self.temperature == 0:
                    # For zero temperature, put all probability on the best move
                    best_move = max(move_counts.items(), key=lambda x: x[1])[0]
                    move_probs = {move: 1.0 if move == best_move else 0.0 for move in move_counts.keys()}
                else:
                    # Apply temperature scaling
                    move_probs = {}
                    for move, count in move_counts.items():
                        if self.temperature < 0.01:
                            # Very low temperature - approximate with best move
                            move_probs[move] = 0.0
                        else:
                            move_probs[move] = (count ** (1.0 / self.temperature)) / total_visits
                    
                    # Normalize if needed
                    prob_sum = sum(move_probs.values())
                    if prob_sum > 0:
                        move_probs = {m: p / prob_sum for m, p in move_probs.items()}
            else:
                # Fallback to uniform probabilities if no visits
                move_probs = {move: 1.0 / len(move_counts) for move in move_counts.keys()}
                
            # Get the best move based on temperature
            if self.temperature == 0:
                # Zero temperature - choose the move with the highest visit count
                best_child = max(root.children, key=lambda c: c.visits)
                selected_move = best_child.move
            else:
                # Apply temperature scaling to visit counts
                visits = np.array([child.visits for child in root.children])
                
                # Prevent numerical issues with very small temperatures
                if self.temperature < 0.01:
                    best_idx = np.argmax(visits)
                    selected_move = root.children[best_idx].move
                else:
                    # Apply temperature scaling
                    scaled_visits = visits ** (1.0 / self.temperature)
                    visit_probs = scaled_visits / scaled_visits.sum()
                    
                    # Choose move randomly according to the temperature-scaled probabilities
                    chosen_idx = np.random.choice(len(root.children), p=visit_probs)
                    selected_move = root.children[chosen_idx].move
                    
            return selected_move, move_probs
        else:
            # Standard game play mode
            if self.temperature == 0:
                # Zero temperature - choose the move with the highest visit count
                best_child = max(root.children, key=lambda c: c.visits)
                return best_child.move
            else:
                # Apply temperature scaling to visit counts
                visits = np.array([child.visits for child in root.children])
                
                # Prevent numerical issues with very small temperatures
                if self.temperature < 0.01:
                    best_idx = np.argmax(visits)
                    return root.children[best_idx].move
                    
                # Apply temperature scaling
                scaled_visits = visits ** (1.0 / self.temperature)
                visit_probs = scaled_visits / scaled_visits.sum()
                
                # Choose move randomly according to the temperature-scaled probabilities
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

    # Add an LRU cache to the evaluate method to avoid recalculating identical positions
    @lru_cache(maxsize=1024)
    def _evaluate_cached(self, board_str):
        """Cached evaluation function for board states."""
        # We can't cache the board object directly as it's not hashable
        # This function is called by evaluate with a string representation of the board
        return self.nn.predict_target(self.board_state_cache[board_str])
    
    def evaluate(self, board):
        """Evaluate a board position, with caching for efficiency."""
        if board.game_over():
            winner = board.get_winner()
            value_vector = np.zeros(3)
            if winner is not None:
                # Assign diminishing values based on player distance from winner
                for p in range(3):
                    distance = (winner - p) % 3  # Distance from winner (0=winner, 1=next, 2=furthest)
                    value_vector[p] = max(0, 1.0 - distance * 0.5)  # 1.0, 0.5, 0.0 for winner, next, furthest
            return value_vector
            
        # Create a cache key based on the board state
        board_str = str(sorted(board.board.items()))
        
        # Check if this state is in our evaluation cache
        if board_str in self.evaluation_cache:
            return self.evaluation_cache[board_str]
        
        # If not in cache, prepare for evaluation
        nn_input = board_to_nn_input(board)
        
        # Store the board state for the cached function
        if not hasattr(self, 'board_state_cache'):
            self.board_state_cache = {}
        self.board_state_cache[board_str] = nn_input
        
        # Get evaluation from the cached function
        _, value = self._evaluate_cached(board_str)
        
        # Store in evaluation cache
        self.evaluation_cache[board_str] = value
        return value
        
    def evaluate_batch(self, boards):
        """Evaluate multiple board positions at once for efficiency."""
        # Split into terminal and non-terminal boards
        terminal_values = []
        non_terminal_boards = []
        non_terminal_indices = []
        
        for i, board in enumerate(boards):
            if board.game_over():
                winner = board.get_winner()
                value_vector = np.zeros(3)
                if winner is not None:
                    # Assign diminishing values based on player distance from winner
                    for p in range(3):
                        distance = (winner - p) % 3  # Distance from winner (0=winner, 1=next, 2=furthest)
                        value_vector[p] = max(0, 1.0 - distance * 0.5)  # 1.0, 0.5, 0.0 for winner, next, furthest
                terminal_values.append((i, value_vector))
            else:
                non_terminal_boards.append(board)
                non_terminal_indices.append(i)
        
        # Process non-terminal boards in batch
        if non_terminal_boards:
            inputs = np.array([board_to_nn_input(b) for b in non_terminal_boards])
            _, values = self.nn.predict_target(inputs, batch=True)
            
            # Update evaluation cache
            for idx, board in enumerate(non_terminal_boards):
                board_str = str(sorted(board.board.items()))
                self.evaluation_cache[board_str] = values[idx]
        
        # Combine results
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
