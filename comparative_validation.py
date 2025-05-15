import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict
import argparse
import glob
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import psutil  # Add psutil for memory tracking

# Import game modules
from training import Board
from model import NeuralNetwork, MCTS
from training import ModelVersion
from validation import PureMCTS, MaxNAlgorithm, ParanoidAlgorithm

# Constants
NEURAL_MCTS_SIMULATIONS = 200  
PURE_MCTS_SIMULATIONS = 200    
MAXN_DEPTH = 3                
PARANOID_DEPTH = 3            
TEMPERATURE = 0.1             
MOVE_DELAY = 0.0              

# Game state wrapper to match the interface expected by validation algorithms
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
        # Convert to a one-hot vector for compatibility
        valid_moves = np.zeros(self.board.cells.__len__())
        for coord in valid_moves_coords:
            idx = self.board.cells.index(coord)
            valid_moves[idx] = 1
        return valid_moves
        
    def make_move(self, action):
        # Convert action index to coordinate
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
        # Count discs for each player
        counts = {0: 0, 1: 0, 2: 0}
        for v in self.board.board.values():
            if v in counts:
                counts[v] += 1
                
        # Determine the player with the most discs
        max_count = max(counts.values())
        rewards = [0.0, 0.0, 0.0]
        
        # Handle ties - multiple players with the same max count
        winners = [p for p, c in counts.items() if c == max_count]
        if len(winners) == 1:
            # Single winner
            winner = winners[0]
            rewards[winner] = 1.0
        else:
            # Tie between multiple players
            for winner in winners:
                rewards[winner] = 1.0 / len(winners)
        
        return rewards

# Add Random algorithm class
class RandomAlgorithm:
    def __init__(self, game, action_threshold=0.0):
        self.game = game
        self.positions_evaluated = 1  # Always 1 for random moves
        self.action_threshold = action_threshold
        
    def search(self, state):
        valid_moves = state.get_valid_moves()
        valid_indices = [i for i, v in enumerate(valid_moves) if v == 1]
        
        if not valid_indices:
            return -1, 1  # No valid moves
            
        # Probabilistically do nothing if random value is below threshold
        if random.random() < self.action_threshold:
            return -1, 1 # Indicate no move was made (pass turn)
            
        # Choose a random valid move
        action = random.choice(valid_indices)
        return action, 1  # Return 1 PPS for consistency

class ComparativeValidation:
    def __init__(self, master):
        self.master = master
        self.master.title("3-Player Hex Othello - Comparative Validation")
        self.master.geometry("400x180")  # Further reduced size
        
        # Add memory tracking variables
        self.memory_usage = []
        self.process = psutil.Process()
        
        # Create frame for model selection
        model_frame = ttk.LabelFrame(master, text="Model Selection")
        model_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Model version selection - use current directory automatically
        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.version_var = tk.StringVar()
        self.version_dropdown = ttk.Combobox(model_frame, textvariable=self.version_var, width=40)
        self.version_dropdown.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(model_frame, text="Refresh", command=self.refresh_versions).grid(row=0, column=2, padx=5, pady=5)
        self.refresh_versions()
        
        # Scenario configuration - games per scenario is now fixed
        # scenario_frame = ttk.LabelFrame(master, text="Scenarios (0 = Skip)") # Frame removed
        # scenario_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.scenarios = [
            "Model vs Pure MCTS vs MaxN",
            "Model vs Pure MCTS vs Paranoid",
            "Model vs Random vs Random"
        ]
        
        # self.games_per_scenario = [] # No longer needed as Tkinter Vars
        
        # Removed loop that created Entry widgets for games per scenario
        # for i, scenario in enumerate(self.scenarios):
        #     ttk.Label(scenario_frame, text=f"Scenario {i+1}: {scenario}").grid(row=i, column=0, sticky="w", padx=5, pady=5)
        #     games_var = tk.StringVar(value="5") # Default was 5, now fixed to 100
        #     games_entry = ttk.Entry(scenario_frame, textvariable=games_var, width=10)
        #     games_entry.grid(row=i, column=1, padx=5, pady=5)
        #     self.games_per_scenario.append(games_var)
        
        # Buttons
        button_frame = ttk.Frame(master)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(button_frame, text="Run Validation", command=self.run_validation).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Exit", command=master.destroy).pack(side="right", padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(master, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", padx=10, pady=5)
    
    def refresh_versions(self):
        """Refresh the list of available model versions from current directory"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_files = glob.glob(os.path.join(current_dir, "*.h5"))
        
        versions = ["Base Model (Untrained)"]  # Always include the base model option
        
        # Extract version numbers from filenames
        for file in model_files:
            base = os.path.basename(file)
            versions.append(base)
        
        self.version_dropdown['values'] = versions
        if versions:
            self.version_dropdown.current(0)
            
        print(f"Found {len(model_files)} model files in current directory")
    
    def run_validation(self):
        """Run the validation with selected parameters"""
        try:
            # Use fixed parameters for simulation parameters
            neural_sims = NEURAL_MCTS_SIMULATIONS
            pure_sims = PURE_MCTS_SIMULATIONS
            maxn_depth = MAXN_DEPTH
            paranoid_depth = PARANOID_DEPTH
            temperature = TEMPERATURE
            move_delay = MOVE_DELAY
            
            # Games per scenario is fixed to 100
            num_scenarios = len(self.scenarios)
            games_counts = [100] * num_scenarios
            
            # Get model version
            selected_version = self.version_var.get()
            
            # Create a separate thread for validation to keep UI responsive
            self.status_var.set("Starting validation...")
            self.master.update()
            
            validation_thread = threading.Thread(target=self.perform_validation_thread, 
                                              args=(selected_version, neural_sims, pure_sims,
                                                    maxn_depth, paranoid_depth, temperature,
                                                    move_delay, games_counts))
            validation_thread.daemon = True
            validation_thread.start()
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please check your inputs: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    
    def perform_validation_thread(self, *args):
        """Thread wrapper for perform_validation to keep UI responsive"""
        try:
            self.perform_validation(*args)
        except Exception as e:
            # Use after() to schedule UI updates from non-UI thread
            self.master.after(0, lambda: messagebox.showerror("Validation Error", f"An error occurred: {e}"))
            self.master.after(0, lambda: self.status_var.set(f"Error: {e}"))
    
    def track_memory_usage(self):
        """Track current memory usage of the process"""
        try:
            memory_info = self.process.memory_info()
            return {
                'rss': memory_info.rss / (1024 * 1024),  # RSS in MB
                'vms': memory_info.vms / (1024 * 1024),  # VMS in MB
                'percent': self.process.memory_percent()
            }
        except:
            return {'rss': 0, 'vms': 0, 'percent': 0}

    def perform_validation(self, selected_version, neural_sims, pure_sims, maxn_depth, paranoid_depth, 
                        temperature, move_delay, games_counts):
        """Perform the actual validation"""
        # Clear previous memory usage data
        self.memory_usage = []
        scenario_memory_usage = {1: [], 2: []}  # Track memory usage for scenarios 1 and 2
        
        # Track initial memory usage
        self.memory_usage.append({
            'timestamp': time.time(),
            'stage': 'start',
            'memory': self.track_memory_usage()
        })

        # Load model
        neural_net = NeuralNetwork()
        
        if selected_version == "Base Model (Untrained)":
            # Use the untrained model
            print("Using untrained base model")
        else:
            # Get model path from current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, selected_version)
            try:
                neural_net.load(model_path)
                print(f"Loaded model: {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                # Update status from background thread
                self.master.after(0, lambda: messagebox.showwarning("Model Loading Error", 
                                               f"Could not load model: {e}\nContinuing with untrained model."))
        
        # Track memory after loading model
        self.memory_usage.append({
            'timestamp': time.time(),
            'stage': 'model_loaded',
            'memory': self.track_memory_usage()
        })

        # Create game instance
        game = Game()
        
        # Create algorithm instances
        # For Neural MCTS, we need a wrapper to adapt the API
        class NeuralMCTSWrapper:
            def __init__(self, mcts_obj):
                self.mcts = mcts_obj
                self.positions_evaluated = 0
                
            def search(self, state):
                # Extract the board from the state
                board = state.board
                player = state.current_player
                
                # Start timing
                start_time = time.time()
                
                # Call the MCTS search
                move = self.mcts.search(board, player)
                
                # Calculate search time
                search_time = time.time() - start_time
                
                # Positions evaluated is roughly proportional to simulations
                self.positions_evaluated = self.mcts.simulations
                pps = self.positions_evaluated / max(0.001, search_time)
                
                if move is None:
                    return -1, pps  # No valid move
                
                # Convert the coordinate move to an action index
                try:
                    action = board.cells.index(move)
                    return action, pps
                except ValueError:
                    return -1, pps  # Move not found in cells
        
        neural_mcts = NeuralMCTSWrapper(MCTS(neural_net, simulations=neural_sims, temperature=temperature))
        pure_mcts = PureMCTS(game, simulations=pure_sims)
        maxn = MaxNAlgorithm(game, max_depth=maxn_depth)
        paranoid = ParanoidAlgorithm(game, max_depth=paranoid_depth)
        # Instantiate RandomAlgorithm with an action_threshold, e.g., 0.25 for 25% chance to pass
        random_algo = RandomAlgorithm(game, action_threshold=0.25)
        
        # Run scenarios
        results = []
        win_rates_data = []
        game_lengths = []
        positions_per_second_data = [] # This stores PPS for all algos in each scenario
        neural_mcts_move_times = [] # New: store avg move time for Neural MCTS for 1st and 2nd scenario only
        active_scenarios = []
        
        for scenario_idx, num_games in enumerate(games_counts):
            if num_games <= 0:
                print(f"Skipping scenario {scenario_idx + 1}")
                continue
                
            active_scenarios.append(self.scenarios[scenario_idx])
            self.status_var.set(f"Running scenario {scenario_idx + 1}: {self.scenarios[scenario_idx]}")
            self.master.update()
            
            if scenario_idx == 0:
                # Scenario 1: Model vs Pure MCTS vs MaxN
                algorithms = [neural_mcts, pure_mcts, maxn]
                algorithm_names = ["Neural MCTS", "Pure MCTS", "MaxN"]
            elif scenario_idx == 1:
                # Scenario 2: Model vs Pure MCTS vs Paranoid
                algorithms = [neural_mcts, pure_mcts, paranoid]
                algorithm_names = ["Neural MCTS", "Pure MCTS", "Paranoid"]
            elif scenario_idx == 2:
                # Scenario 3: Model vs Random vs Random
                algorithms = [neural_mcts, random_algo, random_algo]
                algorithm_names = ["Neural MCTS", "Random 1", "Random 2"]
            
            # Run tournament
            wins, avg_length, pps, avg_move_times = self.run_tournament(game, algorithms, algorithm_names, num_games, move_delay)
            results.append((wins, avg_length, pps))
            
            # Calculate win rates
            win_rates = [w / num_games * 100 for w in wins]
            win_rates_data.append({"names": algorithm_names, "rates": win_rates})
            game_lengths.append(avg_length)
            positions_per_second_data.append({"names": algorithm_names, "pps": pps})

            # Collect Neural MCTS avg move time for 1st and 2nd scenario only
            if scenario_idx in [0, 1] and "Neural MCTS" in algorithm_names:
                nm_idx = algorithm_names.index("Neural MCTS")
                neural_mcts_move_times.append(avg_move_times[nm_idx])
            elif scenario_idx in [0, 1]:
                neural_mcts_move_times.append(0)
            
            # Track memory after scenario completion
            current_memory = self.track_memory_usage()
            self.memory_usage.append({
                'timestamp': time.time(),
                'stage': f'scenario_{scenario_idx + 1}_complete',
                'memory': current_memory
            })
            
            # Store memory usage for scenarios 1 and 2
            if scenario_idx in [0, 1]:  # Scenarios 1 and 2
                scenario_memory_usage[scenario_idx + 1].append(current_memory)
        
        # Calculate and print average memory usage for scenarios 1 and 2
        print("\nMemory Usage Statistics:")
        for scenario_num in [1, 2]:
            if scenario_memory_usage[scenario_num]:
                avg_rss = sum(m['rss'] for m in scenario_memory_usage[scenario_num]) / len(scenario_memory_usage[scenario_num])
                avg_vms = sum(m['vms'] for m in scenario_memory_usage[scenario_num]) / len(scenario_memory_usage[scenario_num])
                print(f"Scenario {scenario_num}:")
                print(f"  Average RSS Memory: {avg_rss:.2f} MB")
                print(f"  Average VMS Memory: {avg_vms:.2f} MB")

        # Plot results (without memory usage plot)
        if active_scenarios:
            self.status_var.set("Plotting results...")
            self.master.update()
            self.plot_results(active_scenarios, win_rates_data, game_lengths, 
                           positions_per_second_data,
                           neural_mcts_move_times,
                           neural_sims, pure_sims, maxn_depth, paranoid_depth)
            self.status_var.set("Validation complete. Results saved to comparative_validation_results.png")
        else:
            self.status_var.set("No scenarios were run.")
    
    def display_board(self, board, title="Board State"):
        """Display a visual representation of the board."""
        # Determine the board size from the cells
        coords = list(board.board.keys())
        q_vals = [q for q, r in coords]
        r_vals = [r for q, r in coords]
        min_q, max_q = min(q_vals), max(q_vals)
        min_r, max_r = min(r_vals), max(r_vals)
        
        # Create a string representation
        board_str = [title + "\n"]
        board_str.append("-" * 40 + "\n")
        
        # Color symbols for each player
        symbols = {-1: ".", 0: "R", 1: "W", 2: "B"}
        
        # Count pieces
        counts = {0: 0, 1: 0, 2: 0}
        for v in board.board.values():
            if v in counts:
                counts[v] += 1
        
        # board_str.append(f"Piece count: Red: {counts[0]}, White: {counts[1]}, Black: {counts[2]}\n") # Removed piece count from console
        board_str.append("-" * 40 + "\n")
        
        # Create visual board representation
        for r in range(min_r, max_r + 1):
            # Indent based on row
            indent = " " * (r - min_r)
            line = indent
            
            for q in range(min_q, max_q + 1):
                if (q, r) in board.board:
                    line += symbols[board.board[(q, r)]] + " "
                else:
                    line += "  "  # Not a valid cell
            
            board_str.append(line + "\n")
        
        # Display in console
        print("".join(board_str))
        
        # Also update status with counts
        self.status_var.set(f"Final board - R: {counts[0]}, W: {counts[1]}, B: {counts[2]}")
        self.master.update()
        
        return "".join(board_str)

    def run_tournament(self, game, algorithms, algorithm_names, num_games, move_delay=0.0):
        """Run a tournament between the algorithms and return the results."""
        wins = [0] * len(algorithms)
        ties = [0] * len(algorithms)
        total_moves = 0
        avg_positions_per_second = [0] * len(algorithms)
        total_move_times = [0.0] * len(algorithms) # To calculate average move times
        game_counts_for_algo = [0] * len(algorithms) # To count games played by each algo slot for avg time
        
        print(f"=== Running tournament: {' vs '.join(algorithm_names)} ===")
        
        for game_idx in range(num_games):
            # Rotate starting player
            starting_player = game_idx % 3
            print(f"Game {game_idx + 1}/{num_games}, starting player: {algorithm_names[starting_player]}")
            
            # Update status
            self.status_var.set(f"Playing game {game_idx + 1}/{num_games} ({algorithm_names[starting_player]} starts)")
            self.master.update()
            
            # Play the game
            rewards, move_count, pps, final_state, game_move_times = self.play_game(game, algorithms, algorithm_names, starting_player, move_delay)
            
            # Show the final board for the first game
            # if game_idx == 0: # Removed final board display from console
            #     self.display_board(final_state.board, title=f"Final Board - Game 1")
            #     
            #     # Brief pause to allow viewing the board
            #     time.sleep(2.0)
            
            # Update statistics
            total_moves += move_count
            for i in range(len(algorithms)):
                avg_positions_per_second[i] += pps[i]
                # Accumulate total move time for each algorithm slot
                # game_move_times is a list of total time spent by each player slot in this game
                total_move_times[i] += game_move_times[i]
                game_counts_for_algo[i] += 1 # Each algo slot plays one game here
                
            # Determine winner(s)
            max_reward = max(rewards)
            winners = [i for i, r in enumerate(rewards) if r == max_reward]
            if len(winners) == 1:
                # Single winner
                winner = winners[0]
                wins[winner] += 1
                print(f"  Winner: {algorithm_names[winner]} with score {rewards}")
            else:
                # Tie between multiple players
                for winner in winners:
                    ties[winner] += 1 / len(winners)  # Split the win credit
                print(f"  Tie game between {', '.join([algorithm_names[w] for w in winners])} with scores {rewards}")
        
        # Calculate averages
        avg_game_length = total_moves / num_games
        avg_algo_move_times = [0.0] * len(algorithms)
        for i in range(len(algorithms)):
            avg_positions_per_second[i] /= num_games
            if game_counts_for_algo[i] > 0 : # num_games would be more direct here if all algos play all games
                 avg_algo_move_times[i] = total_move_times[i] / num_games # Avg time per game for this algo slot
        
        # Print tournament results
        print("\nTournament Results:")
        for i in range(len(algorithms)):
            win_rate = (wins[i] + ties[i]) / num_games * 100
            # Removed avg PPS and avg Move Time from this print:
            print(f"{algorithm_names[i]}: {wins[i]} wins, {ties[i]:.1f} ties ({win_rate:.1f}%)")
        print(f"Average game length: {avg_game_length:.1f} moves\n")
        
        # For consistency with existing code, combine wins and fractional ties
        combined_wins = [wins[i] + ties[i] for i in range(len(algorithms))]
        return combined_wins, avg_game_length, avg_positions_per_second, avg_algo_move_times
    
    def play_game(self, game, algorithms, algorithm_names, starting_player=0, move_delay=0.0):
        """Play a single game with the given algorithms and return the results."""
        state = game.get_initial_state()
        state.current_player = starting_player
        
        move_count = 0
        positions_per_second = [0] * len(algorithms)
        algo_total_thinking_time = [0.0] * len(algorithms)
        algo_move_counts = [0] * len(algorithms)
        
        # Don't update status for every move - we're hiding the gaming process
        # self.status_var.set(f"Player {current_player_idx+1} ({algorithm_names[current_player_idx]}) thinking...")
        # self.master.update()
        
        # Continue until game is terminal
        while not state.is_terminal():
            current_player_idx = state.current_player
            algorithm = algorithms[current_player_idx]
            
            # Get best move from algorithm
            start_think_time = time.time()
            action, pps = algorithm.search(state)
            end_think_time = time.time()
            
            think_time = end_think_time - start_think_time
            algo_total_thinking_time[current_player_idx] += think_time
            algo_move_counts[current_player_idx] += 1
            
            positions_per_second[current_player_idx] += pps
            
            # Make the move
            moved = False
            if action != -1:
                moved = state.make_move(action)
                
            if not moved:
                # No valid moves, pass turn
                state.current_player = (state.current_player + 1) % 3
            
            move_count += 1
            
            # Don't update status for every move - we're hiding the gaming process
            # self.status_var.set(f"Move {move_count}: Player {current_player_idx+1} ({algorithm_names[current_player_idx]})")
            # self.master.update()
            
            # Prevent infinite games
            if move_count > 100:
                break
        
        # Get final rewards
        rewards = state.get_reward()
        
        # Calculate average positions per second for each algorithm
        for i in range(len(algorithms)):
            if algo_move_counts[i] > 0:
                positions_per_second[i] = positions_per_second[i] / algo_move_counts[i]
            else:
                positions_per_second[i] = 0
        
        return rewards, move_count, positions_per_second, state, algo_total_thinking_time
    
    def plot_results(self, scenarios, win_rates_data, game_lengths, positions_per_second_data, neural_mcts_move_times,
                    neural_sims, pure_sims, maxn_depth, paranoid_depth):
        """Plot and save the validation results."""
        N = len(scenarios)
        if N == 0:
            print("No scenarios were run, so no plots will be generated.")
            return

        # Adjust figsize based on the number of scenarios
        fig_height = N * 4.5 + 9  # Removed extra height for memory plot
        plt.figure(figsize=(14, fig_height))

        for i in range(N):
            scenario_label = f"Scenario {i+1}: {scenarios[i]}"
            
            # --- Win Rates for Scenario i ---
            plt.subplot(N + 3, 2, 2 * i + 1)
            win_data = win_rates_data[i]
            algo_names_win = win_data['names']
            rates = win_data['rates']
            bars_win = plt.bar(algo_names_win, rates, color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'][:len(algo_names_win)])
            plt.ylabel('Win Rate (%)')
            plt.title(f"Win Rates\n({scenario_label})")
            plt.xticks(rotation=15, ha="right")
            plt.ylim(0, 100)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            for bar in bars_win:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}%', ha='center', va='bottom' if height > 5 else 'top')

            # --- Average PPS for Scenario i ---
            plt.subplot(N + 3, 2, 2 * i + 2)
            pps_data = positions_per_second_data[i]
            algo_names_pps = pps_data['names']
            pps_values = pps_data['pps']
            bars_pps = plt.bar(algo_names_pps, pps_values, color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'][:len(algo_names_pps)])
            plt.ylabel('Average PPS')
            plt.title(f"Average Positions Per Second (PPS)\n({scenario_label})")
            plt.xticks(rotation=15, ha="right")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            min_pps_val = min(pps_values) if pps_values else 0
            max_pps_val = max(pps_values) if pps_values else 10
            plt.ylim(bottom=0, top=max_pps_val * 1.15 if max_pps_val > 0 else 10)
            for bar in bars_pps:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}', ha='center', va='bottom' if height > (min_pps_val * 0.1) else 'top')
        
        # --- Average Move Time for Neural MCTS (1st and 2nd scenario only) ---
        ax_mt = plt.subplot(N + 3, 1, N + 1) # Row after scenario rows, single plot
        x_mt = [0, 1]
        move_times = neural_mcts_move_times + [0] * (2 - len(neural_mcts_move_times)) # pad if needed
        bars_mt = plt.bar(x_mt, move_times[:2], color='orange')
        plt.ylabel('Avg Move Time (s)')
        plt.title('Neural MCTS Average Move Time (Scenarios 1 & 2)')
        plt.xticks(x_mt, [f"Scenario {i+1}" for i in x_mt], rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars_mt:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom')

        # --- Average Game Length by Scenario (across all scenarios) ---
        ax_gl = plt.subplot(N + 3, 1, N + 2) # Last row, single plot
        x_indices = np.arange(N)
        bars_gl = plt.bar(x_indices, game_lengths, color='teal')
        plt.ylabel('Average Moves')
        plt.title('Average Game Length by Scenario')
        plt.xticks(x_indices, [f"Scenario {i+1}" for i in range(N)], rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars_gl:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}', ha='center', va='bottom')

        plt.suptitle(f"Comparative Validation Results\n(NN sims: {neural_sims}, Pure MCTS sims: {pure_sims}, MaxN depth: {maxn_depth}, Paranoid depth: {paranoid_depth})", fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0, w_pad=2.0)
        
        # Save with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"comparative_validation_{timestamp}.png"
        plt.savefig(filename)
        print(f"Results saved to {filename}")
        plt.close()

def main():
    root = tk.Tk()
    app = ComparativeValidation(root)
    root.mainloop()

if __name__ == "__main__":
    main() 