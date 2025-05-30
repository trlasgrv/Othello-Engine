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
import psutil 
import scipy.stats as stats
import pandas as pd
import json

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels package not installed. Post-hoc analysis will not be available.")
    print("To install statsmodels, run: pip install statsmodels")

from training import Board
from model import NeuralNetwork, MCTS
from training import ModelVersion
from validation import PureMCTS, MaxNAlgorithm, ParanoidAlgorithm

NEURAL_MCTS_SIMULATIONS = 250  
PURE_MCTS_SIMULATIONS = 10    
MAXN_DEPTH = 1                
PARANOID_DEPTH = 1            
TEMPERATURE = 0.1             
MOVE_DELAY = 0.0              

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
        counts = {0: 0, 1: 0, 2: 0}
        for v in self.board.board.values():
            if v in counts:
                counts[v] += 1
                
        max_count = max(counts.values())
        rewards = [0.0, 0.0, 0.0]
        
        winners = [p for p, c in counts.items() if c == max_count]
        if len(winners) == 1:
            winner = winners[0]
            rewards[winner] = 1.0
        else:
            for winner in winners:
                rewards[winner] = 1.0 / len(winners)
        
        return rewards

class RandomAlgorithm:
    def __init__(self, game, action_threshold=0.0):
        self.game = game
        self.positions_evaluated = 1  
        self.action_threshold = action_threshold
        
    def search(self, state):
        valid_moves = state.get_valid_moves()
        valid_indices = [i for i, v in enumerate(valid_moves) if v == 1]
        
        if not valid_indices:
            return -1, 1  
            
        if random.random() < self.action_threshold:
            return -1, 1 
            
        action = random.choice(valid_indices)
        return action, 1 

class ComparativeValidation:
    def __init__(self, master):
        self.master = master
        self.master.title("3-Player Hex Othello - Comparative Validation")
        self.master.geometry("400x180")  
        
        self.memory_usage = []
        self.process = psutil.Process()
        
        model_frame = ttk.LabelFrame(master, text="Model Selection")
        model_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.version_var = tk.StringVar()
        self.version_dropdown = ttk.Combobox(model_frame, textvariable=self.version_var, width=40)
        self.version_dropdown.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(model_frame, text="Refresh", command=self.refresh_versions).grid(row=0, column=2, padx=5, pady=5)
        self.refresh_versions()
        
        
        self.scenarios = [
            "Model vs Pure MCTS vs MaxN",
            "Model vs Pure MCTS vs Paranoid",
            "Model vs Random vs Random"
        ]
        
        
        
        button_frame = ttk.Frame(master)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(button_frame, text="Run Validation", command=self.run_validation).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Exit", command=master.destroy).pack(side="right", padx=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(master, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", padx=10, pady=5)
    
    def refresh_versions(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_files = glob.glob(os.path.join(current_dir, "*.h5"))
        
        versions = ["Base Model (Untrained)"]  
        
        for file in model_files:
            base = os.path.basename(file)
            versions.append(base)
        
        self.version_dropdown['values'] = versions
        if versions:
            self.version_dropdown.current(0)
            
        print(f"Found {len(model_files)} model files in current directory")
    
    def run_validation(self):
        try:
            neural_sims = NEURAL_MCTS_SIMULATIONS
            pure_sims = PURE_MCTS_SIMULATIONS
            maxn_depth = MAXN_DEPTH
            paranoid_depth = PARANOID_DEPTH
            temperature = TEMPERATURE
            move_delay = MOVE_DELAY
            
            num_scenarios = len(self.scenarios)
            games_counts = [50] * num_scenarios
            
            selected_version = self.version_var.get()
            
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
        try:
            self.perform_validation(*args)
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Validation Error", f"An error occurred: {e}"))
            self.master.after(0, lambda: self.status_var.set(f"Error: {e}"))
    
    def track_memory_usage(self):
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
        self.memory_usage = []
        scenario_memory_usage = {1: [], 2: []}  
        
        self.memory_usage.append({
            'timestamp': time.time(),
            'stage': 'start',
            'memory': self.track_memory_usage()
        })

        neural_net = NeuralNetwork()
        
        if selected_version == "Base Model (Untrained)":
            print("Using untrained base model")
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, selected_version)
            try:
                neural_net.load(model_path)
                print(f"Loaded model: {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.master.after(0, lambda: messagebox.showwarning("Model Loading Error", 
                                               f"Could not load model: {e}\nContinuing with untrained model."))
        
        self.memory_usage.append({
            'timestamp': time.time(),
            'stage': 'model_loaded',
            'memory': self.track_memory_usage()
        })

        game = Game()
        
        class NeuralMCTSWrapper:
            def __init__(self, mcts_obj):
                self.mcts = mcts_obj
                self.positions_evaluated = 0
                
            def search(self, state):
                board = state.board
                player = state.current_player
                
                start_time = time.time()
                
                move = self.mcts.search(board, player)
                
                search_time = time.time() - start_time
                
                self.positions_evaluated = self.mcts.simulations
                pps = self.positions_evaluated / max(0.001, search_time)
                
                if move is None:
                    return -1, pps  
                
                try:
                    action = board.cells.index(move)
                    return action, pps
                except ValueError:
                    return -1, pps  
        
        neural_mcts = NeuralMCTSWrapper(MCTS(neural_net, simulations=neural_sims, temperature=temperature))
        pure_mcts = PureMCTS(game, simulations=pure_sims)
        maxn = MaxNAlgorithm(game, max_depth=maxn_depth)
        paranoid = ParanoidAlgorithm(game, max_depth=paranoid_depth)
        random_algo = RandomAlgorithm(game, action_threshold=0.25)
        
        results = []
        win_rates_data = []
        game_lengths = []
        positions_per_second_data = [] 
        neural_mcts_move_times = [] 
        active_scenarios = []
        
        if not hasattr(self, 'raw_data_collection'):
            self.raw_data_collection = []
        
        for scenario_idx, num_games in enumerate(games_counts):
            if num_games <= 0:
                print(f"Skipping scenario {scenario_idx + 1}")
                continue
                
            active_scenarios.append(self.scenarios[scenario_idx])
            self.status_var.set(f"Running scenario {scenario_idx + 1}: {self.scenarios[scenario_idx]}")
            self.master.update()
            
            if scenario_idx == 0:
                algorithms = [neural_mcts, pure_mcts, maxn]
                algorithm_names = ["Neural MCTS", "Pure MCTS", "MaxN"]
            elif scenario_idx == 1:
                algorithms = [neural_mcts, pure_mcts, paranoid]
                algorithm_names = ["Neural MCTS", "Pure MCTS", "Paranoid"]
            elif scenario_idx == 2:
                algorithms = [neural_mcts, random_algo, random_algo]
                algorithm_names = ["Neural MCTS", "Random 1", "Random 2"]
            
            wins, avg_length, pps, avg_move_times, raw_data = self.run_tournament(game, algorithms, algorithm_names, num_games, move_delay)
            results.append((wins, avg_length, pps))
            
            win_rates = [w / num_games * 100 for w in wins]
            win_rates_data.append({"names": algorithm_names, "rates": win_rates})
            game_lengths.append(avg_length)
            positions_per_second_data.append({"names": algorithm_names, "pps": pps})

            if scenario_idx in [0, 1] and "Neural MCTS" in algorithm_names:
                nm_idx = algorithm_names.index("Neural MCTS")
                neural_mcts_move_times.append(avg_move_times[nm_idx])
            elif scenario_idx in [0, 1]:
                neural_mcts_move_times.append(0)
            
            current_memory = self.track_memory_usage()
            self.memory_usage.append({
                'timestamp': time.time(),
                'stage': f'scenario_{scenario_idx + 1}_complete',
                'memory': current_memory
            })
            
            if scenario_idx in [0, 1]:  
                scenario_memory_usage[scenario_idx + 1].append(current_memory)
            
            self.raw_data_collection.append({
                'scenario': scenario_idx + 1,
                'algorithm_names': algorithm_names,
                'individual_results': raw_data['individual_results'],
                'game_lengths': raw_data['game_lengths']
            })
        
        print("\nMemory Usage Statistics:")
        for scenario_num in [1, 2]:
            if scenario_memory_usage[scenario_num]:
                avg_rss = sum(m['rss'] for m in scenario_memory_usage[scenario_num]) / len(scenario_memory_usage[scenario_num])
                avg_vms = sum(m['vms'] for m in scenario_memory_usage[scenario_num]) / len(scenario_memory_usage[scenario_num])
                print(f"Scenario {scenario_num}:")
                print(f"  Average RSS Memory: {avg_rss:.2f} MB")
                print(f"  Average VMS Memory: {avg_vms:.2f} MB")

        self.export_data_for_analysis()

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
        coords = list(board.board.keys())
        q_vals = [q for q, r in coords]
        r_vals = [r for q, r in coords]
        min_q, max_q = min(q_vals), max(q_vals)
        min_r, max_r = min(r_vals), max(r_vals)
        
        board_str = [title + "\n"]
        board_str.append("-" * 40 + "\n")
        
        symbols = {-1: ".", 0: "R", 1: "W", 2: "B"}
        
        counts = {0: 0, 1: 0, 2: 0}
        for v in board.board.values():
            if v in counts:
                counts[v] += 1
        
        board_str.append("-" * 40 + "\n")
        
        for r in range(min_r, max_r + 1):
            indent = " " * (r - min_r)
            line = indent
            
            for q in range(min_q, max_q + 1):
                if (q, r) in board.board:
                    line += symbols[board.board[(q, r)]] + " "
                else:
                    line += "  "  # not a valid cell
            
            board_str.append(line + "\n")
        
        print("".join(board_str))
        
        self.status_var.set(f"Final board - R: {counts[0]}, W: {counts[1]}, B: {counts[2]}")
        self.master.update()
        
        return "".join(board_str)

    def run_tournament(self, game, algorithms, algorithm_names, num_games, move_delay=0.0):
        wins = [0] * len(algorithms)
        ties = [0] * len(algorithms)
        total_moves = 0
        avg_positions_per_second = [0] * len(algorithms)
        total_move_times = [0.0] * len(algorithms)
        game_counts_for_algo = [0] * len(algorithms)
        
        individual_results = [[] for _ in range(len(algorithms))]
        game_lengths_list = []
        
        print(f"=== Running tournament: {' vs '.join(algorithm_names)} ===")
        
        for game_idx in range(num_games):
            starting_player = game_idx % 3
            print(f"Game {game_idx + 1}/{num_games}, starting player: {algorithm_names[starting_player]}")
            
            self.status_var.set(f"Playing game {game_idx + 1}/{num_games} ({algorithm_names[starting_player]} starts)")
            self.master.update()
            
            rewards, move_count, pps, final_state, game_move_times = self.play_game(game, algorithms, algorithm_names, starting_player, move_delay)
            
            total_moves += move_count
            game_lengths_list.append(move_count)  
            
            for i in range(len(algorithms)):
                avg_positions_per_second[i] += pps[i]
                total_move_times[i] += game_move_times[i]
                game_counts_for_algo[i] += 1
                
            max_reward = max(rewards)
            winners = [i for i, r in enumerate(rewards) if r == max_reward]
            
            for i in range(len(algorithms)):
                if i in winners:
                    if len(winners) == 1:
                        individual_results[i].append(1.0)  # Win
                    else:
                        individual_results[i].append(0.5)  # Tie
                else:
                    individual_results[i].append(0.0)  # Loss
                
            if len(winners) == 1:
                winner = winners[0]
                wins[winner] += 1
                print(f"  Winner: {algorithm_names[winner]} with score {rewards}")
            else:
                for winner in winners:
                    ties[winner] += 1 / len(winners)
                print(f"  Tie game between {', '.join([algorithm_names[w] for w in winners])} with scores {rewards}")
        
        avg_game_length = total_moves / num_games
        avg_algo_move_times = [0.0] * len(algorithms)
        for i in range(len(algorithms)):
            avg_positions_per_second[i] /= num_games
            if game_counts_for_algo[i] > 0:
                avg_algo_move_times[i] = total_move_times[i] / num_games
        
        print("\nTournament Results:")
        for i in range(len(algorithms)):
            win_rate = (wins[i] + ties[i]) / num_games * 100
            print(f"{algorithm_names[i]}: {wins[i]} wins, {ties[i]:.1f} ties ({win_rate:.1f}%)")
        print(f"Average game length: {avg_game_length:.1f} moves\n")
        
        self.perform_anova_analysis(algorithm_names, individual_results)
        
        combined_wins = [wins[i] + ties[i] for i in range(len(algorithms))]
        
        return combined_wins, avg_game_length, avg_positions_per_second, avg_algo_move_times, {
            'individual_results': individual_results,
            'game_lengths': game_lengths_list
        }
    
    def play_game(self, game, algorithms, algorithm_names, starting_player=0, move_delay=0.0):
        state = game.get_initial_state()
        state.current_player = starting_player
        
        move_count = 0
        positions_per_second = [0] * len(algorithms)
        algo_total_thinking_time = [0.0] * len(algorithms)
        algo_move_counts = [0] * len(algorithms)
        
        
        while not state.is_terminal():
            current_player_idx = state.current_player
            algorithm = algorithms[current_player_idx]
            
            start_think_time = time.time()
            action, pps = algorithm.search(state)
            end_think_time = time.time()
            
            think_time = end_think_time - start_think_time
            algo_total_thinking_time[current_player_idx] += think_time
            algo_move_counts[current_player_idx] += 1
            
            positions_per_second[current_player_idx] += pps
            
            moved = False
            if action != -1:
                moved = state.make_move(action)
                
            if not moved:
                state.current_player = (state.current_player + 1) % 3
            
            move_count += 1
            
            if move_count > 100:
                break
        
        rewards = state.get_reward()
        
        for i in range(len(algorithms)):
            if algo_move_counts[i] > 0:
                positions_per_second[i] = positions_per_second[i] / algo_move_counts[i]
            else:
                positions_per_second[i] = 0
        
        return rewards, move_count, positions_per_second, state, algo_total_thinking_time
    
    def create_statistical_plot(self, ax, algorithm_names, individual_results, scenario_name):
        data = []
        labels = []
        
        for i, algo_name in enumerate(algorithm_names):
            if individual_results[i]:
                data.append(individual_results[i])
                labels.append(algo_name)
        
        if not data:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title(f"Statistical Comparison\n({scenario_name})")
            return
        
        bp = ax.boxplot(data, patch_artist=True, labels=labels)
        
        colors = ['#88CCEE', '#44AA99', '#CC6677', '#DDCC77', '#AA4499', '#882255']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
        
        for i, d in enumerate(data):
            x = np.random.normal(i+1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=0.3, s=20, c=colors[i % len(colors)])
        
        for i, d in enumerate(data):
            if d:
                mean_val = sum(d) / len(d)
                ax.text(i+1, max(d) + 0.05, f"{mean_val:.2f}", 
                        ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f"Statistical Comparison\n({scenario_name})")
        ax.set_ylabel('Win Rate')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        if len(data) == 2 and data[0] and data[1]:
            t_stat, p_val = stats.ttest_ind(data[0], data[1], equal_var=False)
            p_text = f"p = {p_val:.4f}"
            if p_val < 0.05:
                p_text += " *"
                y = max(max(data[0]), max(data[1])) + 0.1
                ax.plot([1, 2], [y, y], 'k-')
                
            ax.text(1.5, max(max(data[0]), max(data[1])) + 0.15, p_text, 
                    ha='center', va='bottom')
            
    def plot_results(self, scenarios, win_rates_data, game_lengths, positions_per_second_data, neural_mcts_move_times,
                    neural_sims, pure_sims, maxn_depth, paranoid_depth):
        N = len(scenarios)
        if N == 0:
            print("No scenarios were run, so no plots will be generated.")
            return

        fig_height = N * 6.5 + 9  
        plt.figure(figsize=(14, fig_height))

        for i in range(N):
            scenario_label = f"Scenario {i+1}: {scenarios[i]}"
            
            plt.subplot(N * 2 + 3, 2, 2 * i + 1)
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

            plt.subplot(N * 2 + 3, 2, 2 * i + 2)
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
                         
            ax_stat = plt.subplot(N * 2 + 3, 2, 2 * (i + N) + 1)
            
            if hasattr(self, 'raw_data_collection') and i < len(self.raw_data_collection):
                raw_data = self.raw_data_collection[i]
                self.create_statistical_plot(
                    ax_stat, 
                    raw_data['algorithm_names'],
                    raw_data['individual_results'],
                    scenario_label
                )
            else:
                ax_stat.text(0.5, 0.5, "No statistical data available", ha='center', va='center')
                ax_stat.set_title(f"Statistical Comparison\n({scenario_label})")
        
        ax_mt = plt.subplot(N * 2 + 3, 1, N * 2 + 1) 
        x_mt = [0, 1]
        move_times = neural_mcts_move_times + [0] * (2 - len(neural_mcts_move_times)) 
        bars_mt = plt.bar(x_mt, move_times[:2], color='orange')
        plt.ylabel('Avg Move Time (s)')
        plt.title('Neural MCTS Average Move Time (Scenarios 1 & 2)')
        plt.xticks(x_mt, [f"Scenario {i+1}" for i in x_mt], rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars_mt:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom')

        ax_gl = plt.subplot(N * 2 + 3, 1, N * 2 + 2) 
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
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"comparative_validation_{timestamp}.png"
        plt.savefig(filename)
        print(f"Results saved to {filename}")
        plt.close()

    def perform_anova_analysis(self, algorithm_names, individual_results):
        # 2 algorithms with data for ANOVA
        if len(algorithm_names) < 2:
            print("ANOVA analysis requires at least 2 algorithms to compare.")
            return
        
        data = []
        for algo_idx, algo_name in enumerate(algorithm_names):
            for result in individual_results[algo_idx]:
                data.append({
                    'Algorithm': algo_name,
                    'Result': result
                })
        
        df = pd.DataFrame(data)
        
        print("\nDescriptive Statistics:")
        desc_stats = df.groupby('Algorithm')['Result'].agg(['count', 'mean', 'std'])
        print(desc_stats)
        
        groups = [df[df['Algorithm'] == name]['Result'] for name in algorithm_names]
        
        all_same = all(g.std() == 0 for g in groups if len(g) > 0)
        if all_same:
            print("\nCannot perform ANOVA: No variance in at least one group")
            return
        
        try:
            # ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            print("\nANOVA Analysis Results:")
            print(f"F-statistic: {f_stat:.4f}")
            print(f"p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print("There is a statistically significant difference between algorithms")
                
                if STATSMODELS_AVAILABLE:
                    tukey = pairwise_tukeyhsd(df['Result'], df['Algorithm'], alpha=0.05)
                    print("\nTukey HSD Test Results:")
                    print(tukey)
                else:
                    print("\nPost-hoc analysis not available. Please install statsmodels to perform Tukey HSD test.")
                    print("To install statsmodels, run: pip install statsmodels")
                    
                    print("\nPairwise t-test results (p-values):")
                    for i in range(len(algorithm_names)):
                        for j in range(i+1, len(algorithm_names)):
                            group1 = df[df['Algorithm'] == algorithm_names[i]]['Result']
                            group2 = df[df['Algorithm'] == algorithm_names[j]]['Result']
                            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                            print(f"{algorithm_names[i]} vs {algorithm_names[j]}: p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
            else:
                print("No statistically significant difference between algorithms")
        except Exception as e:
            print(f"\nError during statistical analysis: {e}")
            print("This might be due to insufficient data or other statistical issues.")
            print("Basic win rate comparison will still be available in the results.")

    def export_data_for_analysis(self):
        
        if not hasattr(self, 'raw_data_collection') or not self.raw_data_collection:
            print("No data to export.")
            return
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"comparative_validation_data_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.raw_data_collection, f, indent=2)
            print(f"Raw data exported to {filename}")
        except Exception as e:
            print(f"Error exporting data: {e}")

def analyze_saved_data(data_file=None):

    if data_file is None or not os.path.exists(data_file):
        data_files = [f for f in os.listdir(".") if f.endswith("_validation_data.json")]
        
        if not data_files:
            print("No validation data files found.")
            return False
            
        print("Available data files:")
        for i, file in enumerate(data_files):
            print(f"{i+1}. {file}")
            
        try:
            selection = int(input("Enter file number to analyze (0 to exit): "))
            if selection == 0:
                return False
            data_file = data_files[selection - 1]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return False
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
            
        print(f"Analyzing data from: {data_file}")
        print(f"File contains {len(data)} scenarios.")
        
        for scenario in data:
            scenario_num = scenario.get('scenario', 'Unknown')
            algorithms = scenario.get('algorithm_names', [])
            results = scenario.get('individual_results', [])
            
            print(f"\n===== Scenario {scenario_num} =====")
            print(f"Algorithms: {', '.join(algorithms)}")
            
            temp = ComparativeValidation.__new__(ComparativeValidation)
            temp.perform_anova_analysis(algorithms, results)
            
        return True
        
    except Exception as e:
        print(f"Error analyzing data file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Comparative Validation for 3-Player Hex Othello')
    parser.add_argument('--analyze', '-a', metavar='FILE', help='Analyze a saved data file')
    args = parser.parse_args()
    
    if args.analyze:
        analyze_saved_data(args.analyze)
    else:
        root = tk.Tk()
        app = ComparativeValidation(root)
        root.mainloop()

if __name__ == "__main__":
    main() 