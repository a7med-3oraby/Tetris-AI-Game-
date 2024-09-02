# import random
# import numpy as np
# import copy
# import tetris_base_file_v2 as game
# import tetris_ai_file_v2 as ai
# import matplotlib.pyplot as plt
# np.random.seed(42)

# #To store score of best two chromosome.
# best_scores = [[] for _ in range(2)]

# # Initialize chromosome.
# def initialize_chromosome(num_weights, lb=-10, ub=10):
#     chromosome = np.random.uniform(lb, ub, size=(num_weights))
#     return chromosome

# # Calculate fitness(scores) from game_state which is in index 2 of game_state.
# def calculate_fitness(weights, game_state):
#     return game_state[2]

# #calculate best movement to maximize score.
# def calc_best_move(weights, board, piece, show_game=False):
#     best_X = 0
#     best_R = 0
#     best_Y = 0
#     best_score = -100000

#     num_holes_bef, num_blocking_blocks_bef = game.calc_initial_move_info(board)
    
#     for r in range(len(game.PIECES[piece['shape']])):
#         for x in range(-2, game.BOARDWIDTH - 2):
#             movement_info = game.calc_move_info(board, piece, x, r, num_holes_bef, num_blocking_blocks_bef)
 
#             if movement_info[0]:
#                 movement_score = sum(weights[i] * movement_info[i+1] for i in range(len(movement_info)-1))
                
#                 if movement_score > best_score:
#                     best_score = movement_score
#                     best_X = x
#                     best_R = r
#                     best_Y = piece['y']

#     if show_game:
#         piece['y'] = best_Y
#     else:
#         piece['y'] = -2

#     piece['x'] = best_X
#     piece['rotation'] = best_R

#     return best_X, best_R

# # Initialize population(set of chromosomes) where pop_size = 12.
# def initialize_population(num_pop, num_weights=7):
#     population = []
#     for _ in range(num_pop):
#         chromosome = initialize_chromosome(num_weights)
#         population.append(chromosome)
#     return population

# # Evaluate fitness(scores) of each chromosome.
# def evaluate_population(population):
#     evaluations = []
#     for chromosome in population:
#         game_state = ai.run_game(chromosome, 1000000, 100000, False)
#         score = calculate_fitness(chromosome, game_state)
#         evaluations.append((chromosome, score))
#     return evaluations

# # Selection chromosomes as parents for next generation.
# def selection(evaluations, num_selection, type="roulette"):
#     if type == "roulette":
#         return _roulette(evaluations, num_selection)
#     else:
#         raise ValueError(f"Selection type {type} not defined")

# #using roulette to apply selection. 
# def _roulette(evaluations, num_selection):
#     fitness = np.array([eval[1] for eval in evaluations])
#     norm_fitness = fitness / fitness.sum()
#     roulette_prob = np.cumsum(norm_fitness)
#     selected_chromos = []
#     while len(selected_chromos) < num_selection:
#         pick = random.random()
#         for index, (chromosome, *_) in enumerate(evaluations):
#             if pick < roulette_prob[index]:
#                 selected_chromos.append(chromosome)
#                 break
#     return selected_chromos

# # Genetic operator (crossover and mutation).
# def operator(chromosomes, crossover="arithmetic", mutation="random", \
#              crossover_rate=0.5, mutation_rate=0.1):
#     new_chromo = _arithmetic_crossover(chromosomes, mutation, \
#                                        crossover_rate, mutation_rate)
#     _mutation(new_chromo, mutation, mutation_rate)
#     return new_chromo

# #apply arithemtic crossover. 
# def _arithmetic_crossover(selected_pop, mutation, cross_rate=0.5, \
#                          mutation_rate=0.1):
#     N_genes = len(selected_pop[0])
#     new_chromo = [copy.deepcopy(c) for c in selected_pop]

#     for i in range(0, len(selected_pop), 2):
#         if random.random() < cross_rate:
#             try:
#                 a = random.random()
#                 for j in range(0, N_genes):
#                     new_chromo[i][j] = a * new_chromo[i][j] + (1 - a) * new_chromo[i + 1][j]
#                     new_chromo[i + 1][j] = a * new_chromo[i + 1][j] + (1 - a) * new_chromo[i][j]
#             except IndexError:
#                 pass
#     return new_chromo

# #apply random mutation.
# def _mutation(chromosome, type, mutation_rate=0.1):
#     for chromo in chromosome:
#         for i, point in enumerate(chromo):
#             if random.random() < mutation_rate:
#                 chromo[i] = random.uniform(-1.0, 1.0)

# #replace old chromo with new.
# def replace(old_pop, new_chromo):
#     new_pop = sorted(old_pop, key=lambda x: x[1], reverse=True)
#     new_pop[-len(new_chromo):] = new_chromo
#     random.shuffle(new_pop)   #shuffle to maintain diversity.
#     return new_pop

# #main(collecting steps of genatic algorithm including generations & iterations)
# def main_program():
#     # Required settings
#     NUM_POP = 12
#     NUM_GEN = 3
#     TRAIN_ITERATION = 40

#     population = initialize_population(NUM_POP)
#     print(population)

#     best_chromosome = None
#     best_score = float('-inf')

#     for generation in range(NUM_GEN):
#         print(f"Generation {generation + 1}")
        
#         pop = copy.deepcopy(population)          # Initialize population
#         evaluations = evaluate_population(pop)   # Evaluate population


#         # Evolution process within each generation
#         for iteration in range(TRAIN_ITERATION):
#             print(f"Iteration { iteration + 1}")
#             parents = selection(evaluations, len(pop))   # Select parents.
#             offspring = operator(parents)                # Generate offspring.
#             pop = replace(pop, offspring)                # Replace old population with offspring.
#             evaluations = evaluate_population(pop)       # Calculate fitness of the new population.
#             log_values(generation ,iteration , evaluations , "iterations")   # save chromosomes of each generation.
            
#             # Update best chromosome and score
#             for chromo, score in evaluations:
#                if score > best_score:
#                   best_chromosome = chromo
#                   best_score = score

#             # Store scores of the best two chromosomes
#             sorted_evaluations = sorted(evaluations, key=lambda x: x[1], reverse=True)
#             for i in range(2):
#                 best_scores[i].append(sorted_evaluations[i][1])      
       
#         log_values(generation , None ,evaluations ,"generations")   # save chromosomes of each generation.   
#     save_optimal_values(best_chromosome , best_score)   # Save optimal chromosome over all generation.
#     print("Optimal values saved.")

#     # Plot the progress of the best two chromosomes
#     plot_progress()

# #follow progress of best two chromomoses
# def plot_progress():
#     # Plot the progress of the best two chromosomes
#     plt.plot(best_scores[0], label='Best Chromosome 1')
#     plt.plot(best_scores[1], label='Best Chromosome 2')
#     plt.xlabel('Iteration')
#     plt.ylabel('Score')
#     plt.title('Progress of the Best Two Chromosomes')
#     plt.legend()
#     plt.show()

# #log file to save and to analyze the progress of the algorithm.
# def log_values(generation , iteration, evaluations , type):
#     # Log values to a file or print them
#     # You can log chromosome values and scores for each generation
#     if type == "generations":
#       with open("log.txt", "a") as f:
#          if iteration == None:
#             f.write(f"The final chromosomes after end of generation {generation + 1}:\n")
#             for chromo, score in evaluations:
#               f.write(f"Chromosome: {chromo}, Score: {score}\n")
#          else:     
#            f.write(f"Generation {generation + 1}:\n")
#            for chromo, score in evaluations:
#              f.write(f"Chromosome: {chromo}, Score: {score}\n")
#     else:
#       with open("log.txt", "a") as f:
#          f.write(f"Generation {generation + 1}:\n")
#          if iteration == None:
#             f.write(f"The final chromosomes after end of generation {generation + 1}:\n")
#             for chromo, score in evaluations:
#               f.write(f"Chromosome: {chromo}, Score: {score}\n")
#          else:
#             f.write(f"Iteration {iteration + 1}:\n")
#             for chromo, score in evaluations:
#               f.write(f"Chromosome: {chromo}, Score: {score}\n")          

# #another log file To save optimal weights. 
# def save_optimal_values(chromosome , score):
#     # Save optimal chromosome values to a file or print them
#     # Convert the chromosome values to strings and join them with commas
#     chromosome_str = ', '.join(map(str, chromosome))
#     score_str = str(score)  # Convert score to a string
#     with open("optimal_values.txt", "w") as f:
#         f.write("Optimal Chromosome:\n")
#         f.write(str(chromosome_str))
#         f.write("\n")
#         f.write("With Optimal score:\n")
#         f.write(str(score_str))  

# # Run main program.
# if __name__ == "__main__":
#     main_program()

import random
import numpy as np
import copy
import tetris_base_file_v2 as game
import tetris_ai_file_v2 as ai
import matplotlib.pyplot as plt

np.random.seed(42)

#To store score of best two chromosome.
best_scores = [[] for _ in range(2)]

# Initialize chromosome.
def initialize_chromosome(num_weights, lb=-10, ub=10):
    chromosome = np.random.uniform(lb, ub, size=(num_weights))
    return chromosome

# Calculate fitness(scores) from game_state which is in index 2 of game_state.
def calculate_fitness(weights, game_state):
    return game_state[2]

#calculate best movement to maximize score.
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

# Initialize population(set of chromosomes) where pop_size = 12.
def initialize_population(num_pop, num_weights=7):
    population = []
    for _ in range(num_pop):
        chromosome = initialize_chromosome(num_weights)
        population.append(chromosome)
    return population

# Evaluate fitness(scores) of each chromosome.
def evaluate_population(population):
    evaluations = []
    for chromosome in population:
        game_state = ai.run_game(chromosome, 1000000, 100000, False)
        score = calculate_fitness(chromosome, game_state)
        evaluations.append((chromosome, score))
    return evaluations

# Selection chromosomes as parents for next generation.
def selection(evaluations, num_selection, type="roulette"):
    if type == "roulette":
        return _roulette(evaluations, num_selection)
    else:
        raise ValueError(f"Selection type {type} not defined")

#using roulette to apply selection. 
def _roulette(evaluations, num_selection):
    fitness = np.array([eval[1] for eval in evaluations])
    norm_fitness = fitness / fitness.sum()
    roulette_prob = np.cumsum(norm_fitness)
    selected_chromos = []
    while len(selected_chromos) < num_selection:
        pick = random.random()
        for index, (chromosome, *_) in enumerate(evaluations):
            if pick < roulette_prob[index]:
                selected_chromos.append(chromosome)
                break
    return selected_chromos

# Genetic operator (crossover and mutation).
def operator(chromosomes, crossover="arithmetic", mutation="random", \
             crossover_rate=0.5, mutation_rate=0.1):
    new_chromo = _arithmetic_crossover(chromosomes, mutation, \
                                       crossover_rate, mutation_rate)
    _mutation(new_chromo, mutation, mutation_rate)
    return new_chromo

#apply arithemtic crossover. 
def _arithmetic_crossover(selected_pop, mutation, cross_rate=0.5, \
                         mutation_rate=0.1):
    N_genes = len(selected_pop[0])
    new_chromo = [copy.deepcopy(c) for c in selected_pop]

    for i in range(0, len(selected_pop), 2):
        if random.random() < cross_rate:
            try:
                a = random.random()
                for j in range(0, N_genes):
                    new_chromo[i][j] = a * new_chromo[i][j] + (1 - a) * new_chromo[i + 1][j]
                    new_chromo[i + 1][j] = a * new_chromo[i + 1][j] + (1 - a) * new_chromo[i][j]
            except IndexError:
                pass
    return new_chromo

#apply random mutation.
def _mutation(chromosome, type, mutation_rate=0.1):
    for chromo in chromosome:
        for i, point in enumerate(chromo):
            if random.random() < mutation_rate:
                chromo[i] = random.uniform(-1.0, 1.0)

#replace old chromo with new.
def replace(old_pop, new_chromo):
    new_pop = sorted(old_pop, key=lambda x: x[1], reverse=True)
    new_pop[-len(new_chromo):] = new_chromo
    random.shuffle(new_pop)   #shuffle to maintain diversity.
    return new_pop

#main(collecting steps of genatic algorithm including generations & iterations)
def main_program():
    # Required settings
    NUM_POP = 12
    NUM_GEN = 1
    TRAIN_ITERATION = 20

    population = initialize_population(NUM_POP)
    print(population)

    best_chromosome = None
    best_score = float('-inf')

    for generation in range(NUM_GEN):
        print(f"Generation {generation + 1}")
        
        pop = copy.deepcopy(population)          # Initialize population
        evaluations = evaluate_population(pop)   # Evaluate population


        # Evolution process within each generation
        for tr in range(TRAIN_ITERATION):
            print(f"Iteration { tr + 1}")
            parents = selection(evaluations, len(pop))   # Select parents.
            offspring = operator(parents)                # Generate offspring.
            pop = replace(pop, offspring)                # Replace old population with offspring.
            evaluations = evaluate_population(pop)       # Calculate fitness of the new population.
            
            # Update best chromosome and score
            for chromo, score in evaluations:
               if score > best_score:
                  best_chromosome = chromo
                  best_score = score

            # Store scores of the best two chromosomes
            sorted_evaluations = sorted(evaluations, key=lambda x: x[1], reverse=True)
            for i in range(2):
                best_scores[i].append(sorted_evaluations[i][1])      
       
        log_values(generation , evaluations)   # save chromosomes of each generation.   
    save_optimal_values(best_chromosome , best_score)   # Save optimal chromosome over all generation.
    print("Optimal values saved.")

    # Plot the progress of the best two chromosomes
    plot_progress()

#follow progress of best two chromomoses
def plot_progress():
    # Plot the progress of the best two chromosomes
    plt.plot(best_scores[0], label='Best Chromosome 1')
    plt.plot(best_scores[1], label='Best Chromosome 2')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Progress of the Best Two Chromosomes')
    plt.legend()
    plt.show()

#log file to save and to analyze the progress of the algorithm.
def log_values(generation, evaluations):
    # Log values to a file or print them
    # You can log chromosome values and scores for each generation
    with open("log.txt", "a") as f:
        f.write(f"Generation {generation + 1}:\n")
        for chromo, score in evaluations:
            f.write(f"Chromosome: {chromo}, Score: {score}\n")

#another log file To save optimal weights. 
def save_optimal_values(chromosome , score):
    # Save optimal chromosome values to a file or print them
    # Convert the chromosome values to strings and join them with commas
    chromosome_str = ', '.join(map(str, chromosome))
    with open("optimal_values.txt", "w") as f:
        f.write("Optimal Chromosome:\n")
        f.write(str(chromosome_str))
        f.write(str(score))  

# Run main program.
if __name__ == "__main__":
    main_program()






























