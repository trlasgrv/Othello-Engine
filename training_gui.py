# training_gui.py

import tkinter as tk
from tkinter import messagebox, filedialog
import os
from model import NeuralNetwork
from training import self_play, ReplayBuffer, ModelVersion, tournament_play
from gui import play_game
import sys
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import glob

def start_training():
    try:
        # Read parameters from GUI entries
        lr = float(learning_rate_entry.get())
        train_epochs = int(epochs_entry.get())
        mcts_simulations_start = int(mcts_simulations_start_entry.get())
        mcts_simulations_end = int(mcts_simulations_end_entry.get())
        exploration_const = float(c_puct_entry.get())
        temperature_start = float(temperature_start_entry.get())
        temperature_end = float(temperature_end_entry.get())
        iterations = int(iterations_entry.get())
        games_per_iter = int(games_entry.get())
        target_update_freq = int(target_update_entry.get())
        replay_capacity = int(replay_capacity_entry.get())
        validation_split = float(validation_split_entry.get())
        use_augmentation = bool(augmentation_var.get())
        use_tournament = bool(tournament_var.get())
        use_best_elo = bool(best_elo_var.get())
        use_temporal = bool(temporal_var.get())
        tournament_games = int(tournament_games_entry.get())
        max_saved_versions = int(max_versions_entry.get())
        tournament_start = int(tournament_start_entry.get())
    except Exception as e:
        messagebox.showerror("Input error", f"Please check your input parameters.\n{e}")
        return

    # select a starting model or create a new one
    selected_model = select_model_dialog()
    
    # Create the model
    net = NeuralNetwork(learning_rate=lr)
    
    # Create a second model for temporal consistency (previous model)
    prev_net = NeuralNetwork(learning_rate=lr)
    
    # Initialize model versioning system
    model_versioning = ModelVersion(base_filename="model", max_versions=max_saved_versions)
    
    if selected_model and os.path.exists(selected_model):
        try:
            net.load(selected_model)
            prev_net.load(selected_model)  # Also load into previous model
            print(f"Loaded {selected_model} as starting model.", flush=True)
        except Exception as e:
            print(f"Error loading model: {e}", flush=True)
            if not messagebox.askyesno("Model Loading Error", 
                                      f"Error loading selected model: {e}\n\nDo you want to continue with a base model?"):
                return
    else:
        print("Using base model for training.", flush=True)

    # Track when to update the previous model (every 3 iterations)
    prev_model_update_freq = 3
    
    # Create a replay buffer with the specified capacity.
    replay_buffer = ReplayBuffer(capacity=replay_capacity)

    # Clear the progress text widget
    progress_text.delete("1.0", tk.END)

    # Calculate simulation schedule based on curriculum learning
    if mcts_simulations_start == mcts_simulations_end:
        # No curriculum, use fixed simulation count
        simulation_schedule = [mcts_simulations_start] * iterations
    else:
        # Linear increase from start to end
        simulation_schedule = [
            int(mcts_simulations_start + (mcts_simulations_end - mcts_simulations_start) * i / (iterations - 1))
            for i in range(iterations)
        ]
        
    # Calculate temperature schedule
    if temperature_start == temperature_end:
        # No temperature annealing, use fixed temperature
        temperature_schedule = [temperature_start] * iterations
    else:
        # Linear decrease from start to end
        temperature_schedule = [
            temperature_start + (temperature_end - temperature_start) * i / (iterations - 1)
            for i in range(iterations)
        ]

    # Arrays to store metrics for plotting
    all_losses = []
    all_policy_accs = []
    all_value_accs = []
    all_val_losses = []
    all_val_policy_accs = []
    all_val_value_accs = []
    iteration_numbers = []

    # Run training iterations.
    for it in range(iterations):
        progress_text.insert(tk.END, f"\n--- Iteration {it+1}/{iterations} ---\n")
        print(f"\n--- Iteration {it+1}/{iterations} ---", flush=True)
        
        # If using best ELO model and we have run at least one tournament
        if use_best_elo and it > 0 and model_versioning.current_version > 0:
            # Get the best model by ELO rating
            best_models = model_versioning.elo.get_all_ratings()
            if best_models:
                best_version = best_models[0][0]  # First element is (version, rating) with highest rating
                
                # Only switch if best version is different from current version
                if best_version != model_versioning.current_version:
                    progress_text.insert(tk.END, f"Loading best ELO model (Version {best_version})...\n")
                    print(f"Loading best ELO model (Version {best_version})...", flush=True)
                    model_versioning.load_version(net, best_version)
                    progress_text.insert(tk.END, f"Switched to model version {best_version} (highest ELO).\n")
                    print(f"Switched to model version {best_version} (highest ELO).", flush=True)
        
        # Get current simulation count and temperature from schedules
        current_simulations = simulation_schedule[it]
        current_temperature = temperature_schedule[it]
        
        progress_text.insert(tk.END, f"Current MCTS simulations: {current_simulations}\n")
        progress_text.insert(tk.END, f"Current temperature: {current_temperature:.2f}\n")
        print(f"Current MCTS simulations: {current_simulations}", flush=True)
        print(f"Current temperature: {current_temperature:.2f}", flush=True)
        
        # For each game in this iteration play a game and add its examples to the replay buffer.
        for g in range(games_per_iter):
            progress_text.insert(tk.END, f"Playing game {g+1}/{games_per_iter}...\n")
            print(f"Playing game {g+1}/{games_per_iter}...", flush=True)
            
            # Temporal consistency:use previous model for 30% of games
            use_prev_model = (g < games_per_iter * 0.3) and it > 0 and use_temporal
            
            if use_prev_model:
                game_model = prev_net
                progress_text.insert(tk.END, f"Using previous model for temporal consistency.\n")
                print("Using previous model for temporal consistency.", flush=True)
            else:
                game_model = net
            
            examples = self_play(game_model, num_games=1, simulations=current_simulations, 
                                epochs=0, c_puct=exploration_const, 
                                train_after=False, use_augmentation=use_augmentation,
                                temperature=current_temperature)
            replay_buffer.add(examples)
            progress_text.insert(tk.END, f"Game {g+1}/{games_per_iter} complete.\n")
            print(f"Game {g+1}/{games_per_iter} complete.", flush=True)
            progress_text.see(tk.END)
        
        # Now train on all aggregated examples from the replay buffer.
        all_examples = replay_buffer.get_all()
        if all_examples:
            import numpy as np
            X = np.array([ex[0] for ex in all_examples])
            Y_policy = np.array([ex[1] for ex in all_examples])
            Y_value = np.array([ex[2] for ex in all_examples])
            history = net.train(X, Y_policy, Y_value, epochs=train_epochs, validation_split=validation_split)
            # Compute average metrics over the epochs.
            loss_list = history.history["loss"]
            policy_acc_list = history.history["policy_accuracy"]
            value_acc_list = history.history["value_accuracy"]
            val_loss_list = history.history["val_loss"]
            val_policy_acc_list = history.history["val_policy_accuracy"]
            val_value_acc_list = history.history["val_value_accuracy"]

            avg_loss = sum(loss_list) / len(loss_list)
            avg_policy_acc = sum(policy_acc_list) / len(policy_acc_list)
            avg_value_acc = sum(value_acc_list) / len(value_acc_list)
            avg_val_loss = sum(val_loss_list) / len(val_loss_list)
            avg_val_policy_acc = sum(val_policy_acc_list) / len(val_policy_acc_list)
            avg_val_value_acc = sum(val_value_acc_list) / len(val_value_acc_list)

            # Store metrics for plotting
            all_losses.append(avg_loss)
            all_policy_accs.append(avg_policy_acc)
            all_value_accs.append(avg_value_acc)
            all_val_losses.append(avg_val_loss)
            all_val_policy_accs.append(avg_val_policy_acc)
            all_val_value_accs.append(avg_val_value_acc)
            iteration_numbers.append(it+1)

            progress_message = (f"Iteration {it+1}: Loss: {avg_loss:.4f}, "
                                f"Policy Acc: {avg_policy_acc:.4f}, Value Acc: {avg_value_acc:.4f}, "
                                f"Val Loss: {avg_val_loss:.4f}, Val Policy Acc: {avg_val_policy_acc:.4f}, "
                                f"Val Value Acc: {avg_val_value_acc:.4f}\n")
            progress_text.insert(tk.END, progress_message)
            print(progress_message, flush=True)
            progress_text.see(tk.END)
        else:
            progress_text.insert(tk.END, "No training examples generated in this iteration.\n")
            print("No training examples generated in this iteration.", flush=True)
        
        # Save as a new version
        version_file = model_versioning.save_version(net)
        progress_text.insert(tk.END, f"Model saved as version {model_versioning.current_version}.\n")
        print(f"Model saved as version {model_versioning.current_version}.", flush=True)
        progress_text.see(tk.END)
        
        # Run tournament if enabled
        if use_tournament and model_versioning.current_version >= 1 and it >= tournament_start:
            progress_text.insert(tk.END, f"Running tournament against previous versions...\n")
            print("Running tournament against previous versions...", flush=True)
            progress_text.see(tk.END)
            
            ratings = tournament_play(net, model_versioning, num_games=tournament_games, 
                                     simulations=current_simulations)
            
            progress_text.insert(tk.END, "Tournament complete. ELO Ratings:\n")
            for version, rating in ratings:
                version_name = "Current" if version == model_versioning.current_version else f"Version {version}"
                progress_text.insert(tk.END, f"{version_name}: {rating:.1f}\n")
            progress_text.see(tk.END)
        
        # Update the target network based on the specified frequency
        if (it + 1) % target_update_freq == 0:
            net.update_target()
            print("Target network updated.", flush=True)
            progress_text.insert(tk.END, "Target network updated.\n")
            progress_text.see(tk.END)
            
        # Update previous model for temporal consistency every few iterations
        if use_temporal and (it + 1) % prev_model_update_freq == 0:
            # Copy weights from current to previous model
            prev_net.model.set_weights(net.model.get_weights())
            prev_net.target_model.set_weights(net.target_model.get_weights())
            print("Previous model updated for temporal consistency.", flush=True)
            progress_text.insert(tk.END, "Previous model updated for temporal consistency.\n")
            progress_text.see(tk.END)
    
    # Create directory for plots if it doesn't exist
    plots_dir = "training_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate individual metric plots
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_numbers, all_losses, 'b-', label='Training Loss')
    plt.plot(iteration_numbers, all_val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plots_dir}/loss_{timestamp}.png")
    
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_numbers, all_policy_accs, 'b-', label='Training Policy Accuracy')
    plt.plot(iteration_numbers, all_val_policy_accs, 'r-', label='Validation Policy Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Policy Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plots_dir}/policy_accuracy_{timestamp}.png")
    
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_numbers, all_value_accs, 'b-', label='Training Value Accuracy')
    plt.plot(iteration_numbers, all_val_value_accs, 'r-', label='Validation Value Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Value Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plots_dir}/value_accuracy_{timestamp}.png")
    
    # Create combined plot
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(iteration_numbers, all_losses, 'b-', label='Training Loss')
    plt.plot(iteration_numbers, all_val_losses, 'r-', label='Validation Loss')
    plt.ylabel('Loss')
    plt.title('Training Metrics')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(iteration_numbers, all_policy_accs, 'b-', label='Training Policy Accuracy')
    plt.plot(iteration_numbers, all_val_policy_accs, 'r-', label='Validation Policy Accuracy')
    plt.ylabel('Policy Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(iteration_numbers, all_value_accs, 'b-', label='Training Value Accuracy')
    plt.plot(iteration_numbers, all_val_value_accs, 'r-', label='Validation Value Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Value Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/combined_metrics_{timestamp}.png")
    
    progress_text.insert(tk.END, f"\nTraining metrics plots saved to {plots_dir}/ directory.\n")
    progress_text.insert(tk.END, "\nTraining complete.\n")
    print(f"\nTraining metrics plots saved to {plots_dir}/ directory.", flush=True)
    print("\nTraining complete.", flush=True)

def get_available_models():
    """Get a list of all available model files in the current directory."""
    # Get all .h5 files in the current directory
    model_files = glob.glob("*.h5")
    # Also get version models
    model_files.extend(glob.glob("model_v*.h5"))
    # Remove duplicates and sort
    model_files = sorted(list(set(model_files)))
    return model_files

def select_model_dialog():
    """Open a dialog to select a model file."""
    models = get_available_models()
    
    # Create a new dialog window
    dialog = tk.Toplevel()
    dialog.title("Select Model")
    dialog.geometry("400x300")
    dialog.transient(root)  # Set to be on top of the main window
    dialog.grab_set()  # Modal dialog
    
    # Add a label
    tk.Label(dialog, text="Select a model to use:", font=("Arial", 12)).pack(pady=10)
    
    # Add a listbox with all models
    listbox = tk.Listbox(dialog, width=50, height=10)
    listbox.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
    
    # Add Use Base Model option at the top
    listbox.insert(tk.END, "Base Model (No File)")
    
    # Add all model files
    for model in models:
        listbox.insert(tk.END, model)
    
    # Select the first item by default
    listbox.selection_set(0)
    
    # Variable to store the selected model
    selected_model = [None]  # Use a list to be able to modify it from the inner function
    
    # Function to handle selection
    def on_select():
        idx = listbox.curselection()
        if idx:
            idx = idx[0]
            if idx == 0:  # Base model option
                selected_model[0] = None
            else:
                selected_model[0] = listbox.get(idx)
        dialog.destroy()
    
    # Function to handle cancel
    def on_cancel():
        selected_model[0] = None
        dialog.destroy()
    
    # Add buttons
    button_frame = tk.Frame(dialog)
    button_frame.pack(pady=10, fill=tk.X)
    
    tk.Button(button_frame, text="Select", command=on_select, width=10).pack(side=tk.LEFT, padx=20)
    tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.RIGHT, padx=20)
    
    # Wait for the dialog to be closed
    dialog.wait_window()
    
    return selected_model[0]

def demonstrate_ai():
    # Let the user select a model
    selected_model = select_model_dialog()
    
    net = NeuralNetwork()
    
    if selected_model and os.path.exists(selected_model):
        try:
            net.load(selected_model)
            print(f"Loaded {selected_model} for demonstration.", flush=True)
        except Exception as e:
            print(f"Error loading model {selected_model} for demonstration: {e}", flush=True)
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return
    else:
        print("Using base model for demonstration.", flush=True)
    
    play_game(net, human_player=None)

def play_as_human():
    # Let the user select a model
    selected_model = select_model_dialog()
    
    net = NeuralNetwork()
    
    if selected_model and os.path.exists(selected_model):
        try:
            net.load(selected_model)
            print(f"Loaded {selected_model} for human play.", flush=True)
        except Exception as e:
            print(f"Error loading model {selected_model} for human play: {e}", flush=True)
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return
    else:
        print("Using base model for human play.", flush=True)
    
    play_game(net, human_player=0)

# Build the training GUI using Tkinter
root = tk.Tk()
root.title("3-Player Hex Othello - Training GUI")

# Learning rate
tk.Label(root, text="Learning Rate:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
learning_rate_entry = tk.Entry(root)
learning_rate_entry.insert(0, "0.0001")
learning_rate_entry.grid(row=0, column=1, padx=5, pady=5)

# Training epochs per iteration
tk.Label(root, text="Epochs per iteration:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
epochs_entry = tk.Entry(root)
epochs_entry.insert(0, "2")
epochs_entry.grid(row=1, column=1, padx=5, pady=5)

# MCTS simulations: starting value
tk.Label(root, text="MCTS Simulations (Start):").grid(row=2, column=0, sticky="e", padx=5, pady=5)
mcts_simulations_start_entry = tk.Entry(root)
mcts_simulations_start_entry.insert(0, "30")
mcts_simulations_start_entry.grid(row=2, column=1, padx=5, pady=5)

# MCTS simulations: ending value (for curriculum learning)
tk.Label(root, text="MCTS Simulations (End):").grid(row=3, column=0, sticky="e", padx=5, pady=5)
mcts_simulations_end_entry = tk.Entry(root)
mcts_simulations_end_entry.insert(0, "80")
mcts_simulations_end_entry.grid(row=3, column=1, padx=5, pady=5)

# Exploration constant (c_puct)
tk.Label(root, text="Exploration Constant (c_puct):").grid(row=4, column=0, sticky="e", padx=5, pady=5)
c_puct_entry = tk.Entry(root)
c_puct_entry.insert(0, "2.0")
c_puct_entry.grid(row=4, column=1, padx=5, pady=5)

# Temperature: starting value
tk.Label(root, text="Temperature (Start):").grid(row=5, column=0, sticky="e", padx=5, pady=5)
temperature_start_entry = tk.Entry(root)
temperature_start_entry.insert(0, "1.5")
temperature_start_entry.grid(row=5, column=1, padx=5, pady=5)

# Temperature: ending value
tk.Label(root, text="Temperature (End):").grid(row=6, column=0, sticky="e", padx=5, pady=5)
temperature_end_entry = tk.Entry(root)
temperature_end_entry.insert(0, "0.5")
temperature_end_entry.grid(row=6, column=1, padx=5, pady=5)

# Number of iterations
tk.Label(root, text="Iterations:").grid(row=7, column=0, sticky="e", padx=5, pady=5)
iterations_entry = tk.Entry(root)
iterations_entry.insert(0, "15")
iterations_entry.grid(row=7, column=1, padx=5, pady=5)

# Number of games per iteration
tk.Label(root, text="Games per iteration:").grid(row=8, column=0, sticky="e", padx=5, pady=5)
games_entry = tk.Entry(root)
games_entry.insert(0, "5")
games_entry.grid(row=8, column=1, padx=5, pady=5)

# Target network update frequency (in iterations)
tk.Label(root, text="Target Update Frequency (iters):").grid(row=9, column=0, sticky="e", padx=5, pady=5)
target_update_entry = tk.Entry(root)
target_update_entry.insert(0, "1")
target_update_entry.grid(row=9, column=1, padx=5, pady=5)

# Replay buffer capacity
tk.Label(root, text="Replay Buffer Capacity:").grid(row=10, column=0, sticky="e", padx=5, pady=5)
replay_capacity_entry = tk.Entry(root)
replay_capacity_entry.insert(0, "2000")
replay_capacity_entry.grid(row=10, column=1, padx=5, pady=5)

# Validation split
tk.Label(root, text="Validation Split:").grid(row=11, column=0, sticky="e", padx=5, pady=5)
validation_split_entry = tk.Entry(root)
validation_split_entry.insert(0, "0.2")
validation_split_entry.grid(row=11, column=1, padx=5, pady=5)

# Use data augmentation
augmentation_var = tk.IntVar(value=1)  # Enabled by default
tk.Checkbutton(root, text="Use Hexagonal Data Augmentation", variable=augmentation_var).grid(row=12, column=0, columnspan=2, sticky="w", padx=5, pady=5)

# Use tournament
tournament_var = tk.IntVar(value=0)
tk.Checkbutton(root, text="Run Tournament", variable=tournament_var).grid(row=13, column=0, columnspan=2, sticky="w", padx=5, pady=5)

# Use best ELO model each iteration
best_elo_var = tk.IntVar(value=0)
tk.Checkbutton(root, text="Use Best ELO Model Each Iteration", variable=best_elo_var).grid(row=14, column=0, columnspan=2, sticky="w", padx=5, pady=5)

# Use temporal consistency (previous model)
temporal_var = tk.IntVar(value=1)  # On by default for stability
tk.Checkbutton(root, text="Use Temporal Consistency (30% Previous Model)", variable=temporal_var).grid(row=15, column=0, columnspan=2, sticky="w", padx=5, pady=5)

# Tournament start iteration
tk.Label(root, text="Tournament Start Iteration:").grid(row=16, column=0, sticky="e", padx=5, pady=5)
tournament_start_entry = tk.Entry(root)
tournament_start_entry.insert(0, "4")
tournament_start_entry.grid(row=16, column=1, padx=5, pady=5)

# Number of tournament games
tk.Label(root, text="Number of Tournament Games:").grid(row=17, column=0, sticky="e", padx=5, pady=5)
tournament_games_entry = tk.Entry(root)
tournament_games_entry.insert(0, "5")
tournament_games_entry.grid(row=17, column=1, padx=5, pady=5)

# Maximum saved versions
tk.Label(root, text="Maximum Saved Versions:").grid(row=18, column=0, sticky="e", padx=5, pady=5)
max_versions_entry = tk.Entry(root)
max_versions_entry.insert(0, "5")
max_versions_entry.grid(row=18, column=1, padx=5, pady=5)

# Buttons for actions
tk.Button(root, text="Start Training", command=start_training).grid(row=19, column=0, columnspan=2, pady=5)
tk.Button(root, text="Demonstrate AI vs AI", command=demonstrate_ai).grid(row=20, column=0, columnspan=2, pady=5)
tk.Button(root, text="Play as Human (Red)", command=play_as_human).grid(row=21, column=0, columnspan=2, pady=5)
tk.Button(root, text="Exit", command=root.quit).grid(row=22, column=0, columnspan=2, pady=5)

# A Text widget to show progress.
progress_text = tk.Text(root, height=12, width=60)
progress_text.grid(row=23, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()
