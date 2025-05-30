import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict

from training import Board
from model import NeuralNetwork, MCTS

NEURAL_MCTS_SIMULATIONS = 200
PURE_MCTS_SIMULATIONS = 200
MAXN_DEPTH = 3
PARANOID_DEPTH = 3
GAMES_PER_SCENARIO = 50

class Game:
    def __init__(self):
        pass
        
    def get_initial_state(self):
        return GameState()

class GameState:
    def __init__(self, board=None):
        self.board = board if board else Board()
        self.current_player = 0
        
    def clone(self):
        new_state = GameState(self.board.copy())
        new_state.current_player = self.current_player
        return new_state
        
    def get_valid_moves(self):
        valid_moves_coords = self.board.get_valid_moves(self.current_player)
        valid_moves = np.zeros(self.board.cells.__len__())
        for coord in valid_moves_coords:
            idx = self.board.cells.index(coord)
            valid_moves[idx] = 1
        return valid_moves
        
    def make_move(self, action):
        if action >= 0 and action < len(self.board.cells):
            coord = self.board.cells[action]
            if coord in self.board.get_valid_moves(self.current_player):
                self.board.apply_move(coord, self.current_player)
                self.current_player = (self.current_player + 1) % 3
                return True
        return False
        
    def is_terminal(self):
        return self.board.game_over()
        
    def get_reward(self):
        winner = self.board.get_winner()
        rewards = [0, 0, 0]
        if winner is not None:
            rewards[winner] = 1
        return rewards

class PureMCTS:
    def __init__(self, game, simulations=PURE_MCTS_SIMULATIONS, c_puct=1.5):
        self.game = game
        self.simulations = simulations
        self.c_puct = c_puct
        self.Q = {}  # Q values 
        self.N = {}  # visit counts
        self.positions_evaluated = 0
        
    def _get_node_key(self, state, action=None):
        board_repr = ""
        for coord in sorted(state.board.board.keys()):
            board_repr += f"{coord}:{state.board.board[coord]},"
        board_repr += f"player:{state.current_player}"
        
        if action is not None:
            return f"{board_repr}:{action}"
        return board_repr
        
    def _select(self, state):
        node_key = self._get_node_key(state)
        valid_moves = state.get_valid_moves()
        
        if not np.any(valid_moves) or state.is_terminal():
            return None
            
        best_score = -float('inf')
        best_action = None
        total_visits = sum(self.N.get(self._get_node_key(state, a), 0) for a in range(len(valid_moves)) if valid_moves[a])
        
        for action in range(len(valid_moves)):
            if valid_moves[action]:
                action_key = self._get_node_key(state, action)
                
                if action_key not in self.N:
                    return action
                    
                # UCB calculation
                q_value = self.Q.get(action_key, 0)
                n_visits = self.N.get(action_key, 0)
                
                ucb_score = q_value + self.c_puct * np.sqrt(total_visits) / (1 + n_visits)
                
                if ucb_score > best_score:
                    best_score = ucb_score
                    best_action = action
                    
        return best_action
    
    def _expand_and_evaluate(self, state):
        self.positions_evaluated += 1
        if state.is_terminal():
            return state.get_reward()
            
        sim_state = state.clone()
        current_player = sim_state.current_player
        
        while not sim_state.is_terminal():
            valid_moves = sim_state.get_valid_moves()
            if not np.any(valid_moves):
                break
                
            actions = [i for i, valid in enumerate(valid_moves) if valid]
            action = random.choice(actions)
            sim_state.make_move(action)
            
        rewards = sim_state.get_reward()
        return rewards
    
    def _backup(self, path, rewards):
        for state, action in path:
            action_key = self._get_node_key(state, action)
            current_player = state.current_player
            
            self.N[action_key] = self.N.get(action_key, 0) + 1
            
            current_q = self.Q.get(action_key, 0)
            new_q = current_q + (rewards[current_player] - current_q) / self.N[action_key]
            self.Q[action_key] = new_q
    
    def search(self, state):
        start_time = time.time()
        self.positions_evaluated = 0
        
        for _ in range(self.simulations):
            sim_state = state.clone()
            path = []
            
            while True:
                action = self._select(sim_state)
                
                if action is None:
                    break
                    
                path.append((sim_state.clone(), action))
                sim_state.make_move(action)
            
            rewards = self._expand_and_evaluate(sim_state)
            
            self._backup(path, rewards)
        
        valid_moves = state.get_valid_moves()
        best_action = -1
        most_visits = -1
        
        for action in range(len(valid_moves)):
            if valid_moves[action]:
                action_key = self._get_node_key(state, action)
                visits = self.N.get(action_key, 0)
                
                if visits > most_visits:
                    most_visits = visits
                    best_action = action
        
        positions_per_second = self.positions_evaluated / (time.time() - start_time) if time.time() > start_time else 0
        return best_action, positions_per_second

class MaxNAlgorithm:
    def __init__(self, game, max_depth=MAXN_DEPTH):
        self.game = game
        self.max_depth = max_depth
        self.positions_evaluated = 0
    
    def search(self, state):
        start_time = time.time()
        self.positions_evaluated = 0
        
        valid_moves = state.get_valid_moves()
        best_action = -1
        best_value = [-float('inf')] * 3  # For 3 players
        
        current_player = state.current_player
        
        for action in range(len(valid_moves)):
            if valid_moves[action]:
                next_state = state.clone()
                next_state.make_move(action)
                
                value = self._maxn(next_state, self.max_depth - 1)
                
                if value[current_player] > best_value[current_player]:
                    best_value = value
                    best_action = action
        
        positions_per_second = self.positions_evaluated / (time.time() - start_time) if time.time() > start_time else 0
        return best_action, positions_per_second
    
    def _maxn(self, state, depth):
        self.positions_evaluated += 1
        
        if depth == 0 or state.is_terminal():
            if state.is_terminal():
                return state.get_reward()
            else:
                return self._evaluate_state(state)
        
        valid_moves = state.get_valid_moves()
        current_player = state.current_player
        
        best_value = [-float('inf')] * 3  # for 3 players
        
        if not np.any(valid_moves):
            next_state = state.clone()
            next_state.current_player = (next_state.current_player + 1) % 3
            return self._maxn(next_state, depth - 1)
        
        for action in range(len(valid_moves)):
            if valid_moves[action]:
                next_state = state.clone()
                next_state.make_move(action)
                
                value = self._maxn(next_state, depth - 1)
                
                if value[current_player] > best_value[current_player]:
                    best_value = value
        
        return best_value
    
    def _evaluate_state(self, state):
        board = state.board
        player_counts = [0, 0, 0]
        
        for coord, player_idx in board.board.items():
            if player_idx >= 0 and player_idx < 3:  
                player_counts[player_idx] += 1
        
        total_pieces = sum(player_counts)
        if total_pieces > 0:
            normalized_scores = [count / total_pieces for count in player_counts]
        else:
            normalized_scores = [1/3, 1/3, 1/3]  
        
        return normalized_scores

class ParanoidAlgorithm:
    def __init__(self, game, max_depth=PARANOID_DEPTH):
        self.game = game
        self.max_depth = max_depth
        self.positions_evaluated = 0
    
    def search(self, state):
        start_time = time.time()
        self.positions_evaluated = 0
        
        valid_moves = state.get_valid_moves()
        best_action = -1
        best_value = -float('inf')
        
        current_player = state.current_player
        
        for action in range(len(valid_moves)):
            if valid_moves[action]:
                next_state = state.clone()
                next_state.make_move(action)
                
                value = self._paranoid(next_state, self.max_depth - 1, current_player)
                
                if value > best_value:
                    best_value = value
                    best_action = action
        
        positions_per_second = self.positions_evaluated / (time.time() - start_time) if time.time() > start_time else 0
        return best_action, positions_per_second
    
    def _paranoid(self, state, depth, original_player):
        self.positions_evaluated += 1
        
        if depth == 0 or state.is_terminal():
            if state.is_terminal():
                rewards = state.get_reward()
                return rewards[original_player]
            else:
                return self._evaluate_state(state, original_player)
        
        valid_moves = state.get_valid_moves()
        current_player = state.current_player
        
        if not np.any(valid_moves):
            next_state = state.clone()
            next_state.current_player = (next_state.current_player + 1) % 3
            return self._paranoid(next_state, depth - 1, original_player)
        
        if current_player == original_player:
            best_value = -float('inf')
            for action in range(len(valid_moves)):
                if valid_moves[action]:
                    next_state = state.clone()
                    next_state.make_move(action)
                    value = self._paranoid(next_state, depth - 1, original_player)
                    best_value = max(best_value, value)
            return best_value
        
        else:
            best_value = float('inf')
            for action in range(len(valid_moves)):
                if valid_moves[action]:
                    next_state = state.clone()
                    next_state.make_move(action)
                    value = self._paranoid(next_state, depth - 1, original_player)
                    best_value = min(best_value, value)
            return best_value
    
    def _evaluate_state(self, state, player):
        board = state.board
        player_count = 0
        opponent_counts = [0, 0]
        
        for coord, cell in board.board.items():
            if cell == player:
                player_count += 1
            elif 0 <= cell < 3:  # valid opponent
                idx = 0 if cell < player else 1
                opponent_counts[idx] += 1
        
        total_pieces = player_count + sum(opponent_counts)
        if total_pieces > 0:
            player_advantage = player_count / total_pieces - sum(opponent_counts) / (2 * total_pieces)
            return player_advantage
        else:
            return 0  

def play_game(game, algorithms, starting_player=0):
    state = game.get_initial_state()
    state.current_player = starting_player
    
    move_count = 0
    positions_per_second = [0] * len(algorithms)
    positions_evaluated = 0
    
    while not state.is_terminal():
        current_player = state.current_player
        algorithm = algorithms[current_player]
        
        action, pps = algorithm.search(state)
        positions_per_second[current_player] += pps
        
        if action != -1:
            state.make_move(action)
        else:
            # no valid moves
            state.current_player = (state.current_player + 1) % 3
        
        move_count += 1
        
        if move_count > 100:
            break
    
    rewards = state.get_reward()
    
    for i in range(len(algorithms)):
        positions_per_second[i] = positions_per_second[i] / max(1, move_count / 3)
    
    return rewards, move_count, positions_per_second

def run_tournament(game, algorithms, algorithm_names, num_games=GAMES_PER_SCENARIO):
    wins = [0] * len(algorithms)
    total_moves = 0
    avg_positions_per_second = [0] * len(algorithms)
    
    print(f"=== Running tournament: {' vs '.join(algorithm_names)} ===")
    
    for game_idx in range(num_games):
        starting_player = game_idx % 3
        print(f"Game {game_idx + 1}/{num_games}, starting player: {algorithm_names[starting_player]}")
        
        rewards, move_count, pps = play_game(game, algorithms, starting_player)
        
        total_moves += move_count
        for i in range(len(algorithms)):
            avg_positions_per_second[i] += pps[i]
            
        winner = np.argmax(rewards)
        wins[winner] += 1
        
        print(f"  Winner: {algorithm_names[winner]} with score {rewards}")
    
    avg_game_length = total_moves / num_games
    for i in range(len(algorithms)):
        avg_positions_per_second[i] /= num_games
    
    print("\nTournament Results:")
    for i in range(len(algorithms)):
        win_rate = wins[i] / num_games * 100
        print(f"{algorithm_names[i]}: {wins[i]} wins ({win_rate:.1f}%), avg PPS: {avg_positions_per_second[i]:.1f}")
    print(f"Average game length: {avg_game_length:.1f} moves\n")
    
    return wins, avg_game_length, avg_positions_per_second

def plot_results(scenarios, win_rates, game_lengths, positions_per_second):
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 2, 1)
    width = 0.25
    x = np.arange(len(scenarios))
    
    for i, algorithm in enumerate(["Neural MCTS", "Pure MCTS", "MaxN", "Paranoid"]):
        algorithm_win_rates = []
        for scenario in win_rates:
            for j, name in enumerate(scenario["names"]):
                if name == algorithm and j < len(scenario["rates"]):
                    algorithm_win_rates.append(scenario["rates"][j])
                    break
            else:
                algorithm_win_rates.append(0)  
        
        plt.bar(x + (i - 1.5) * width, algorithm_win_rates, width, label=algorithm)
    
    plt.xlabel('Scenario')
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rates by Algorithm and Scenario')
    plt.xticks(x, [f"Scenario {i+1}" for i in range(len(scenarios))])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 2)
    plt.bar(x, game_lengths)
    plt.xlabel('Scenario')
    plt.ylabel('Average Moves')
    plt.title('Average Game Length by Scenario')
    plt.xticks(x, [f"Scenario {i+1}" for i in range(len(scenarios))])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 3)
    width = 0.25
    
    for i, algorithm in enumerate(["Neural MCTS", "Pure MCTS", "MaxN", "Paranoid"]):
        algorithm_pps = []
        for scenario in positions_per_second:
            for j, name in enumerate(scenario["names"]):
                if name == algorithm and j < len(scenario["pps"]):
                    algorithm_pps.append(scenario["pps"][j])
                    break
            else:
                algorithm_pps.append(0)  
        
        plt.bar(x + (i - 1.5) * width, algorithm_pps, width, label=algorithm)
    
    plt.xlabel('Scenario')
    plt.ylabel('Positions per Second')
    plt.title('Algorithm Efficiency')
    plt.xticks(x, [f"Scenario {i+1}" for i in range(len(scenarios))])
    plt.yscale('log')  
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 4)
    plt.axis('off')
    scenario_text = "Validation Scenarios:\n\n"
    for i, scenario in enumerate(scenarios):
        scenario_text += f"Scenario {i+1}: {scenario}\n"
        
    scenario_text += "\nValidation Parameters:\n"
    scenario_text += f"- Neural MCTS: {NEURAL_MCTS_SIMULATIONS} simulations\n"
    scenario_text += f"- Pure MCTS: {PURE_MCTS_SIMULATIONS} simulations\n"
    scenario_text += f"- MaxN: depth {MAXN_DEPTH}\n"
    scenario_text += f"- Paranoid: depth {PARANOID_DEPTH}\n"
    scenario_text += f"- Games per scenario: {GAMES_PER_SCENARIO}"
    
    plt.text(0.1, 0.9, scenario_text, va='top', ha='left', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig("validation_results.png")
    plt.show()

def run_validation(model_path="model.h5"):
    print("Starting validation of the 3-Player Hex Othello Engine")
    
    neural_net = NeuralNetwork()
    try:
        neural_net.load(model_path)
        print(f"Loaded {model_path} for validation")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing with untrained model")
    
    game = Game()
    
    neural_mcts = MCTS(game, neural_net, simulations=NEURAL_MCTS_SIMULATIONS)
    pure_mcts = PureMCTS(game)
    maxn = MaxNAlgorithm(game)
    paranoid = ParanoidAlgorithm(game)
    
    scenarios = [
        "Neural MCTS vs Pure MCTS vs MaxN",
        "Neural MCTS vs Pure MCTS vs Paranoid",
        "Neural MCTS vs MaxN vs Paranoid"
    ]
    
    results = []
    win_rates_data = []
    game_lengths = []
    positions_per_second_data = []
    
    # scenario 1
    algorithms = [neural_mcts, pure_mcts, maxn]
    algorithm_names = ["Neural MCTS", "Pure MCTS", "MaxN"]
    wins, avg_length, pps = run_tournament(game, algorithms, algorithm_names)
    results.append((wins, avg_length, pps))
    
    win_rates = [w / GAMES_PER_SCENARIO * 100 for w in wins]
    win_rates_data.append({"names": algorithm_names, "rates": win_rates})
    game_lengths.append(avg_length)
    positions_per_second_data.append({"names": algorithm_names, "pps": pps})
    
    # scenario 2
    algorithms = [neural_mcts, pure_mcts, paranoid]
    algorithm_names = ["Neural MCTS", "Pure MCTS", "Paranoid"]
    wins, avg_length, pps = run_tournament(game, algorithms, algorithm_names)
    results.append((wins, avg_length, pps))
    
    win_rates = [w / GAMES_PER_SCENARIO * 100 for w in wins]
    win_rates_data.append({"names": algorithm_names, "rates": win_rates})
    game_lengths.append(avg_length)
    positions_per_second_data.append({"names": algorithm_names, "pps": pps})
    
    algorithms = [neural_mcts, maxn, paranoid]
    algorithm_names = ["Neural MCTS", "MaxN", "Paranoid"]
    wins, avg_length, pps = run_tournament(game, algorithms, algorithm_names)
    results.append((wins, avg_length, pps))
    
    win_rates = [w / GAMES_PER_SCENARIO * 100 for w in wins]
    win_rates_data.append({"names": algorithm_names, "rates": win_rates})
    game_lengths.append(avg_length)
    positions_per_second_data.append({"names": algorithm_names, "pps": pps})
    
    plot_results(scenarios, win_rates_data, game_lengths, positions_per_second_data)
    print("Validation complete. Results saved to validation_results.png")

if __name__ == "__main__":
    run_validation() 