import random, time, pygame, sys
from pygame.locals import *
import tetris_base_file_v2 as game
import tetris_ai_file_v2 as ai
import numpy as np
import copy
import ga_v2 as ga
size = [640, 480]
screen = pygame.display.set_mode((size[0], size[1]))

def run_game(chromosome, speed, max_score=100000, no_show=False):
    game.FPS = int(speed)
    game.main()

    board = game.get_blank_board()
    last_fall_time = time.time()
    score = 0
    level, fall_freq = game.calc_level_and_fall_freq(score)
    falling_piece = game.get_new_piece()
    next_piece = game.get_new_piece()

    calc_best_move(chromosome, board, falling_piece)
    
    num_used_pieces = 0
    removed_lines = [0, 0, 0, 0]  # Combos

    alive = True
    win = False

    while alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Game exited by user")
                exit()
        
        if falling_piece is None:
            falling_piece = next_piece
            next_piece = game.get_new_piece()

            calc_best_move(chromosome, board, falling_piece,no_show)

            num_used_pieces += 1
            score += 1
        

            last_fall_time = time.time()

            if not game.is_valid_position(board, falling_piece):
                alive = False
        
    

        if no_show or time.time() - last_fall_time > fall_freq:
            if not game.is_valid_position(board, falling_piece, adj_Y=1):
                game.add_to_board(board, falling_piece)

                num_removed_lines = game.remove_complete_lines(board)
                if num_removed_lines == 1:
                    score += 40
                    removed_lines[0] += 1
                elif num_removed_lines == 2:
                    score += 120
                    removed_lines[1] += 1
                elif num_removed_lines == 3:
                    score += 300
                    removed_lines[2] += 1
                elif num_removed_lines == 4:
                    score += 1200
                    removed_lines[3] += 1

                falling_piece = None
            else:
                falling_piece['y'] += 1
                last_fall_time = time.time()
        

        if not no_show:
            draw_game_on_screen(board, score, level, next_piece, falling_piece,
                                chromosome)

        if score > max_score:
            alive = False
            win = True

    game_state = [num_used_pieces, removed_lines, score, win]

    return game_state

def draw_game_on_screen(board, score, level, next_piece, falling_piece, chromosome):
    game.DISPLAYSURF.fill(game.BGCOLOR)
    game.draw_board(board)
    game.draw_status(score, level)
    game.draw_next_piece(next_piece)

    if falling_piece is not None:
        game.draw_piece(falling_piece)

    pygame.display.update()
    game.FPSCLOCK.tick(game.FPS)

def calc_best_move(weights, board, piece, show_game=False):
    best_X = 0
    best_R = 0
    best_Y = 0
    best_score = -100000

    num_holes_bef, num_blocking_blocks_bef = game.calc_initial_move_info(board)
    
    for r in range(len(game.PIECES[piece['shape']])):
        for x in range(-2, game.BOARDWIDTH - 2):
            movement_info = game.calc_move_info(board, piece, x, r, num_holes_bef, num_blocking_blocks_bef)
 
            if movement_info[0]:
                movement_score = sum(weights[i] * movement_info[i+1] for i in range(len(movement_info)-1))
                
                if movement_score > best_score:
                    best_score = movement_score
                    best_X = x
                    best_R = r
                    best_Y = piece['y']

    if show_game:
        piece['y'] = best_Y
    else:
        piece['y'] = -2

    piece['x'] = best_X
    piece['rotation'] = best_R

    return best_X, best_R

   









































