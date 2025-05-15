# 3-Player Hex Othello AI: Training and Validation

This project provides scripts to train a neural network for a 3-player Hex Othello game and to perform comparative validation of trained models against other AI algorithms.

## Setup and Dependencies

Ensure you have Python 3 installed. The main dependencies include:

*   `tensorflow`
*   `numpy`
*   `matplotlib`
*   `psutil`
*   `tkinter` (usually included with Python standard library)

You can typically install these using pip:
`pip install tensorflow numpy matplotlib psutil`

## Training New Models (`training_gui.py`)

1.  **Run the script**:
    ```bash
    python training_gui.py
    ```
2.  **Function**: This opens a GUI that allows you to configure and start the training process for the AI model. You can set parameters like learning rate, number of iterations, MCTS simulations, etc.
3.  **Output**: Trained models are saved as `.h5` files (e.g., `model_v1.h5`, `model_v2.h5`) in the same directory where the script is run. You can also choose to start training from an existing model file.

## Comparative Validation (`comparative_validation.py`)

1.  **Run the script**:
    ```bash
    python comparative_validation.py
    ```
2.  **Function**: This script opens a simple GUI to compare a selected trained model against a set of predefined AI algorithms (Pure MCTS, MaxN, Paranoid, Random).
3.  **Model Selection**:
    *   The GUI will automatically list any `.h5` model files found in the same directory as the script.
    *   You can select one of these models (or a "Base Model (Untrained)") to be the primary model for comparison.
4.  **Scenarios**: The script runs a fixed set of 3 scenarios, each for 100 games:
    *   Scenario 1: Your Model vs Pure MCTS vs MaxN
    *   Scenario 2: Your Model vs Pure MCTS vs Paranoid
    *   Scenario 3: Your Model vs Random vs Random
5.  **Output**:
    *   Console output will show progress and final win/loss/tie statistics for each scenario.
    *   Average memory usage for the first two scenarios will be printed.
    *   A PNG image file (e.g., `comparative_validation_YYYYMMDD-HHMMSS.png`) will be saved in the current directory, containing plots of:
        *   Win rates for each algorithm in each scenario.
        *   Average PPS (Positions Per Second) for each algorithm in each scenario.
        *   Average move time for your selected model (Neural MCTS) in the first two scenarios.
        *   Overall average game length for each scenario.

## Model Files

*   The `.h5` files are the saved neural network models. These are used by both the training script (to continue training or as a base) and the validation script (to evaluate performance).
