import math
import random
import numpy as np
from model import NeuralNetwork, MCTS, board_to_nn_input
import os

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)  
tf.config.threading.set_inter_op_parallelism_threads(2)  

os.environ['TF_ENABLE_CPU_NUMA_AWARENESS'] = '1'
os.environ['TF_ENABLE_MKL_NATIVE_FORMAT'] = '1'


BOARD_RADIUS = 6 

def generate_hex_board(radius):
    cells = []
    for q in range(-radius + 1, radius):
        for r in range(-radius + 1, radius):
            if max(abs(q), abs(r), abs(-q - r)) < radius:
                cells.append((q, r))
    return cells

def rotate_hex_60_degrees(q, r):
    return (-r, q + r)

def rotate_hex_120_degrees(q, r):
    return (-q - r, q)

def rotate_hex_180_degrees(q, r):
    return (-q, -r)

def rotate_hex_240_degrees(q, r):
    return (r, -q - r)

def rotate_hex_300_degrees(q, r):
    return (q + r, -q)

def rotate_board_60_degrees(board_dict):
    new_board = {}
    for (q, r), value in board_dict.items():
        new_q, new_r = rotate_hex_60_degrees(q, r)
        new_board[(new_q, new_r)] = value
    return new_board

def rotate_board_120_degrees(board_dict):
    new_board = {}
    for (q, r), value in board_dict.items():
        new_q, new_r = rotate_hex_120_degrees(q, r)
        new_board[(new_q, new_r)] = value
    return new_board

def rotate_board_180_degrees(board_dict):
    new_board = {}
    for (q, r), value in board_dict.items():
        new_q, new_r = rotate_hex_180_degrees(q, r)
        new_board[(new_q, new_r)] = value
    return new_board

def rotate_board_240_degrees(board_dict):
    new_board = {}
    for (q, r), value in board_dict.items():
        new_q, new_r = rotate_hex_240_degrees(q, r)
        new_board[(new_q, new_r)] = value
    return new_board

def rotate_board_300_degrees(board_dict):
    new_board = {}
    for (q, r), value in board_dict.items():
        new_q, new_r = rotate_hex_300_degrees(q, r)
        new_board[(new_q, new_r)] = value
    return new_board

def augment_nn_input(board_tensor):
    augmented_tensors = [board_tensor]
    
    rot_60 = np.rot90(board_tensor, k=1, axes=(0, 1))
    augmented_tensors.append(rot_60)
    
    rot_120 = np.rot90(board_tensor, k=2, axes=(0, 1))
    augmented_tensors.append(rot_120)
    
    rot_180 = np.rot90(board_tensor, k=3, axes=(0, 1))
    augmented_tensors.append(rot_180)
    
    rot_240 = np.rot90(board_tensor, k=4, axes=(0, 1))
    augmented_tensors.append(rot_240)
    
    rot_300 = np.rot90(board_tensor, k=5, axes=(0, 1))
    augmented_tensors.append(rot_300)
    
    return augmented_tensors

class Board:
    def __init__(self):
        self.cells = generate_hex_board(BOARD_RADIUS)
        self.board = {cell: -1 for cell in self.cells}
        self.first_move_done = {0: False, 1: False, 2: False}
        self.init_board()

    def init_board(self):
        try:
            self.board[(0, -1)] = 0  # Red
            self.board[(1, -1)] = 1  # White
            self.board[(0, 0)]  = 2  # Black
        except KeyError:
            print("Error: Could not place initial discs. Check coordinates.")

    def on_board(self, coord):
        return coord in self.board

    def outflanks_in_direction(self, coord, direction, player):
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
        if self.board[coord] != -1:
            return False
        axial_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        for d in axial_dirs:
            if self.outflanks_in_direction(coord, d, player):
                return True
        return False

    def get_valid_moves(self, player):
        return [c for c in self.cells if self.board[c] == -1 and self.is_valid_move(c, player)]

    def apply_move(self, coord, player):
        if not self.is_valid_move(coord, player):
            raise ValueError("apply_move: invalid move.")
        self.board[coord] = player
        if not self.first_move_done[player]:
            self.first_move_done[player] = True
        else:
            axial_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
            for d in axial_dirs:
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
        for p in [0, 1, 2]:
            if self.get_valid_moves(p):
                return False
        return True

    def get_winner(self):
        counts = {0: 0, 1: 0, 2: 0}
        for v in self.board.values():
            if v in counts:
                counts[v] += 1
        mx = max(counts.values())
        winners = [p for p, c in counts.items() if c == mx]
        if len(winners) == 1:
            return winners[0]
        return None

    def copy(self):
        new_board = Board()
        new_board.board = self.board.copy()
        new_board.first_move_done = self.first_move_done.copy()
        return new_board

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def add(self, examples):
        self.buffer.extend(examples)
        if len(self.buffer) > self.capacity:
            # only the most recent examples
            self.buffer = self.buffer[-self.capacity:]
    
    def get_all(self):
        return self.buffer

class EloRating:
    def __init__(self, k_factor=32, initial_rating=1400):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}  
    
    def get_rating(self, model_version):
        return self.ratings.get(model_version, self.initial_rating)
    
    def add_model(self, model_version, rating=None):
        if rating is None:
            rating = self.initial_rating
        self.ratings[model_version] = rating
    
    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    
    def update_rating_two_player(self, rating_a, rating_b, score_a):
        expected_a = self.expected_score(rating_a, rating_b)
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        return new_rating_a
    
    def update_ratings_three_player(self, version_a, version_b, version_c, results):
        rating_a = self.get_rating(version_a)
        rating_b = self.get_rating(version_b)
        rating_c = self.get_rating(version_c)
        
        score_a, score_b, score_c = results
        
        expected_a_vs_b = self.expected_score(rating_a, rating_b)
        if score_a > score_b:
            outcome_a_vs_b = 1
        elif score_a < score_b:
            outcome_a_vs_b = 0
        else:
            outcome_a_vs_b = 0.5
        
        expected_a_vs_c = self.expected_score(rating_a, rating_c)
        if score_a > score_c:
            outcome_a_vs_c = 1
        elif score_a < score_c:
            outcome_a_vs_c = 0
        else:
            outcome_a_vs_c = 0.5
        
        expected_b_vs_c = self.expected_score(rating_b, rating_c)
        if score_b > score_c:
            outcome_b_vs_c = 1
        elif score_b < score_c:
            outcome_b_vs_c = 0
        else:
            outcome_b_vs_c = 0.5
        
        delta_a = self.k_factor * ((outcome_a_vs_b - expected_a_vs_b) + (outcome_a_vs_c - expected_a_vs_c)) / 2
        delta_b = self.k_factor * ((1 - outcome_a_vs_b - expected_a_vs_b) + (outcome_b_vs_c - expected_b_vs_c)) / 2
        delta_c = self.k_factor * ((1 - outcome_a_vs_c - expected_a_vs_c) + (1 - outcome_b_vs_c - expected_b_vs_c)) / 2
        
        self.ratings[version_a] = rating_a + delta_a
        self.ratings[version_b] = rating_b + delta_b
        self.ratings[version_c] = rating_c + delta_c
        
        return {
            version_a: self.ratings[version_a],
            version_b: self.ratings[version_b],
            version_c: self.ratings[version_c]
        }
    
    def get_all_ratings(self):
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

class ModelVersion:
    def __init__(self, base_filename="model", max_versions=10):
        self.base_filename = base_filename
        self.max_versions = max_versions
        self.current_version = 0
        self.elo = EloRating()
        self.elo.add_model(0)  # Add base model
    
    def save_version(self, neural_net):
        self.current_version += 1
        version_filename = f"{self.base_filename}_v{self.current_version}.h5"
        neural_net.save(version_filename)
        
        neural_net.save(f"{self.base_filename}.h5")
        
        self.elo.add_model(self.current_version)
        
        if self.current_version > self.max_versions:
            old_version = self.current_version - self.max_versions
            old_filename = f"{self.base_filename}_v{old_version}.h5"
            try:
                if os.path.exists(old_filename):
                    os.remove(old_filename)
            except Exception as e:
                print(f"Error removing old model version {old_version}: {e}")
        
        return version_filename
    
    def load_version(self, neural_net, version=None):
        if version is None:
            version = self.current_version
        
        if version == 0:
            return False
        
        version_filename = f"{self.base_filename}_v{version}.h5"
        try:
            if os.path.exists(version_filename):
                neural_net.load(version_filename)
                return True
            else:
                print(f"Model version {version} not found.")
                return False
        except Exception as e:
            print(f"Error loading model version {version}: {e}")
            return False
    
    def select_opponent_versions(self, num_opponents=2):
        available_versions = list(range(max(1, self.current_version - self.max_versions + 1), 
                                     self.current_version + 1))
        
        if not available_versions:
            return [0] * num_opponents  
        
        ratings = [self.elo.get_rating(v) for v in available_versions]
        min_rating = min(ratings)
        adjusted_ratings = [r - min_rating + 1 for r in ratings]  
        total = sum(adjusted_ratings)
        probabilities = [r / total for r in adjusted_ratings]
        
        selected = []
        for _ in range(min(num_opponents, len(available_versions))):
            idx = random.choices(range(len(available_versions)), probabilities)[0]
            selected.append(available_versions[idx])
            available_versions.pop(idx)
            probabilities.pop(idx)
            if not available_versions:
                break
        
        while len(selected) < num_opponents:
            selected.append(0)
        
        return selected

def self_play(neural_net, num_games=10, simulations=50, epochs=1, c_puct=1.0, 
            train_after=True, use_augmentation=True, temperature=1.0):
    training_examples = []
    
    def process_and_augment(state, policy_target, value_target, augment=True):
        if not augment:
            return [(state, policy_target, value_target)]
            
        examples = []
        augmented_states = augment_nn_input(state)
        for aug_state in augmented_states:
            examples.append((aug_state, policy_target, value_target))
        return examples

    for game_idx in range(num_games):
        board = Board()
        current_player = 0
        game_history = []
        
        current_temp = temperature
        temp_decay_rate = 0.85  
        temp_min = 0.2  
        state_policies = {}  
        
        move_counter = 0
        max_moves = 100  
        
        while not board.game_over() and move_counter < max_moves:
            move_counter += 1
            valid_moves = board.get_valid_moves(current_player)
            if not valid_moves:
                current_player = (current_player + 1) % 3
                continue

            mcts = MCTS(neural_net, simulations=simulations, c_puct=c_puct, temperature=current_temp)
            
            state = board_to_nn_input(board)
            
            move, move_probs = mcts.search(board, current_player, return_probs=True)
            
            policy_vector = np.zeros(3)
            
            if move_probs:
                prob_sum = sum(move_probs.values())
                if prob_sum > 0:
                    policy_vector[current_player] = 1.0
            
            policy_sum = np.sum(policy_vector)
            if policy_sum > 0:
                policy_vector = policy_vector / policy_sum
            
            state_key = tuple(state.flatten())
            state_policies[state_key] = policy_vector
            
            game_history.append((state, current_player))
            
            if move is None:
                current_player = (current_player + 1) % 3
                continue
                
            try:
                board.apply_move(move, current_player)
            except Exception as e:
                print("Error applying move in self_play:", e)
                break
                
            current_player = (current_player + 1) % 3
            
            current_temp = max(temp_min, current_temp * temp_decay_rate)

        winner = board.get_winner()
        result = np.zeros(3)
        if winner is not None:
            result[winner] = 0.9
            for p in range(3):
                if p != winner:
                    result[p] = 0.05
        else:
            result = np.ones(3) / 3

        for state, player in game_history:
            state_key = tuple(state.flatten())
            
            if state_key in state_policies:
                policy_target = state_policies[state_key]
                policy_target = policy_target**1.5
                policy_target = policy_target / np.sum(policy_target)
            else:
                policy_target = result.copy()
            
            value_target = result.copy()
            
            new_examples = process_and_augment(state, policy_target, value_target, 
                                             augment=use_augmentation)
            training_examples.extend(new_examples)

    if train_after:
        if training_examples:
            X = np.array([ex[0] for ex in training_examples])
            Y_policy = np.array([ex[1] for ex in training_examples])
            Y_value = np.array([ex[2] for ex in training_examples])
            
            batch_size = 64
            if len(training_examples) > 500:
                batch_size = 128  
                
            history = neural_net.train(X, Y_policy, Y_value, epochs=epochs, batch_size=batch_size)
            return history
        else:
            return None
    else:
        return training_examples

def tournament_play(current_model, model_versioning, num_games=5, simulations=100):
    print(f"Starting tournament with {num_games} games...")
    
    player_current = 0
    
    player1_net = NeuralNetwork()
    player2_net = NeuralNetwork()
    
    for game_idx in range(num_games):
        opponent_versions = model_versioning.select_opponent_versions(num_opponents=2)
        player1_version, player2_version = opponent_versions
        
        if player1_version > 0:
            model_versioning.load_version(player1_net, player1_version)
        if player2_version > 0:
            model_versioning.load_version(player2_net, player2_version)
            
        print(f"Game {game_idx+1}/{num_games}: Current model vs. Version {player1_version} vs. Version {player2_version}")
        
        board = Board()
        current_player = 0
        
        while not board.game_over():
            valid_moves = board.get_valid_moves(current_player)
            if not valid_moves:
                current_player = (current_player + 1) % 3
                continue
                
            if current_player == player_current:
                model = current_model
            elif current_player == 1:
                model = player1_net
            else:
                model = player2_net
                
            mcts = MCTS(model, simulations=simulations)
            move = mcts.search(board, current_player)
            
            if move is None:
                current_player = (current_player + 1) % 3
                continue
                
            try:
                board.apply_move(move, current_player)
            except Exception as e:
                print(f"Error applying move in tournament_play: {e}")
                break
                
            current_player = (current_player + 1) % 3
            
        winner = board.get_winner()
        if winner is not None:
            print(f"Game {game_idx+1} winner: {'Current' if winner == player_current else f'Version {player1_version}' if winner == 1 else f'Version {player2_version}'}")
            results = [0, 0, 0]
            results[winner] = 1
        else:
            print(f"Game {game_idx+1} ended in a tie")
            results = [1/3, 1/3, 1/3]
            
        updated_ratings = model_versioning.elo.update_ratings_three_player(
            model_versioning.current_version + 1,  
            player1_version,
            player2_version,
            results
        )
        
        print(f"Updated ratings: Current: {updated_ratings[model_versioning.current_version + 1]:.1f}, "
              f"Version {player1_version}: {updated_ratings[player1_version]:.1f}, "
              f"Version {player2_version}: {updated_ratings[player2_version]:.1f}")
        
    print("\nFinal ELO Ratings:")
    for version, rating in model_versioning.elo.get_all_ratings():
        version_name = "Current" if version == model_versioning.current_version + 1 else f"Version {version}"
        print(f"{version_name}: {rating:.1f}")
        
    return model_versioning.elo.get_all_ratings()
