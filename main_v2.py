import random
import numpy as np
import copy
import tetris_base_file_v2 as game
import tetris_ai_file_v2 as ai
import analyser as analyser
import matplotlib.pyplot as plt
import argparse
import pdb
import ga_v2 as ga
import tetris_ai_file_v2 as ai

def main():
    TEST_ITERATION = 20
    optimal_values_file = "optimal_values.txt"
    log_file = "test_log.txt"
    
    with open(optimal_values_file, "r") as f:
        optimal_weights = list(map(float, f.readline().split(',')))
    
    with open(log_file, "w") as log:
        for i in range(TEST_ITERATION):
            score = ai.run_game(optimal_weights, speed=1000000, max_score=100000, no_show=False)
            log.write(f"Iteration {i+1}: Score = {score[2]}\n")
            print(f"Iteration {i+1}" )
            

if __name__ == "_main_":
    main()




#-0.10588533098081698, 6.247524755717409, -1.1688451507153168, -0.3907169017251196, 1.3532181849691405, 0.5877770552630885, 1.1414462138956312






















