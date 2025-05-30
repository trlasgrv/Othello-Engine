import sys
import math
import random
import pygame
from pygame import gfxdraw

from training import Board  
from model import MCTS, NeuralNetwork  

BOARD_RADIUS = 6  
HEX_SIZE = 40
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)

player_colors = {
    0: RED,    
    1: WHITE,  
    2: BLACK   
}

BOARD_CELL_FILL = (150, 75, 0)  
BOARD_BORDER_COLOR = (0, 0, 255)  
BG_COLOR = WHITE
TEXT_COLOR = (20, 20, 20)

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
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, pos)

def draw_board(screen, board, cell_positions):
    for coord, center in cell_positions.items():
        corners = polygon_corners(center, HEX_SIZE - 2)
        gfxdraw.aapolygon(screen, corners, BOARD_BORDER_COLOR)
        gfxdraw.filled_polygon(screen, corners, BOARD_CELL_FILL)

        val = board.board[coord]
        if val in [0, 1, 2]:
            disc_color = player_colors[val]
            inner = polygon_corners(center, HEX_SIZE - 12)
            gfxdraw.aapolygon(screen, inner, disc_color)
            gfxdraw.filled_polygon(screen, inner, disc_color)

def precompute_cell_positions(board, origin):
    positions = {}
    for c in board.cells:
        positions[c] = axial_to_pixel(c[0], c[1], HEX_SIZE, origin)
    return positions

def play_game(neural_net, human_player=None):
    board = Board()
    current_player = 0
    origin = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
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
                    elif event.key == pygame.K_e and human_player is not None and current_player == human_player:
                        valid_moves = board.get_valid_moves(current_player)
                        if not valid_moves:
                            current_player = (current_player + 1) % 3
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if human_player is not None and current_player == human_player:
                        pos = pygame.mouse.get_pos()
                        clicked = None
                        for coord, center in cell_positions.items():
                            if math.hypot(center[0] - pos[0], center[1] - pos[1]) < HEX_SIZE:
                                clicked = coord
                                break
                        if clicked and clicked in board.get_valid_moves(current_player):
                            board.apply_move(clicked, current_player)
                            current_player = (current_player + 1) % 3

            # AI move if not human's turn
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
            turn_str = f"Turn: {'Red' if current_player == 0 else 'White' if current_player == 1 else 'Black'}"
            draw_text(screen, turn_str, (10, 10))
            draw_text(screen, "Press R to restart", (10, 40))
            if human_player is not None:
                valid_moves = board.get_valid_moves(current_player)
                if not valid_moves and current_player == human_player:
                    draw_text(screen, "No valid moves. Press E to skip", (10, 70))
                else:
                    draw_text(screen, "Click a cell to play", (10, 70))

            pygame.display.flip()
            clock.tick(5)

            # Check if game over
            if board.game_over():
                winner = board.get_winner()
                if winner is not None:
                    msg = f"Game Over! Winner: {'Red' if winner == 0 else 'White' if winner == 1 else 'Black'}"
                else:
                    msg = "Game Over! It's a tie."
                print(msg)
                screen.fill(BG_COLOR)
                draw_text(screen, msg, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 20), font_size=32)
                pygame.display.flip()
                pygame.time.wait(3000)
                return
        except Exception as e:
            print("Error in game loop:", e)
            pygame.quit()
            sys.exit()
