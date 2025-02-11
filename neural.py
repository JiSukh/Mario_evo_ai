import numpy as np
import torch
import torch.nn as nn
from emu_data_reader import emu_read as er
import random
import subprocess
import threading
import time



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(75, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 3)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output_layer(x)
        x = self.activation(x)
        return x

def run_game():
    # Run Mesen as a subprocess in a separate thread
    command = ["Mesen.exe", "--testrunner", "game\Super Mario Bros (E).nes", "memory_reader.lua", "--emulation.emulationSpeed=5000"]
    subprocess.run(command,)
        
def initialise_population(size):
    population = []
    for i in range(size):
        network = NeuralNetwork()

        for param in network.parameters():
            if len(param.shape) > 1:  # Weight matrices, not biases
                nn.init.xavier_normal_(param)
            else:
                param.data = torch.randn_like(param.data)
        population.append(network)
    return population



def fitness_function(network):
    #"timer", "player_x_screen", "level", "world", "player_lives"
    
    timer_w = 0
    pos_bias = 1.5
    level_bias = 3
    world_bias = 10
    
    score = {}
    fitness = 0
    
    max_pos = 0
  
    game_thread = threading.Thread(target=run_game)
    game_thread.start()
    
    while game_thread.is_alive():
        game_data,score = er.read_game_memory()
        if game_data:
            normalised_data = er.normalise_data(game_data)
            in_tensor = torch.FloatTensor(normalised_data)
            outputs = network(in_tensor)
            er.write_inputs(outputs)
            
            max_pos = max(max_pos, score['player_x_level'])
            
            fitness = max(fitness, (timer_w*score['timer']) + (pos_bias*score['player_x_level']) + (level_bias*score['level']) + (world_bias*score['world']))

            
        time.sleep(0.02)
    if score:
        if score["death_flag"] == 1:
            print("death")
            fitness -= 700
        if score["timeout"] == 1:
            print("time")
            fitness -= 500
        print(f"Fitness: {fitness} - Position: {max_pos}")
    return fitness/5





def crossover(parents):
    parent1, parent2 = parents
    child = NeuralNetwork()
    for param1, param2, child_param in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
        child_param.data = (param1.data + param2.data) / 2
    return child


def mutate(child, mutation_rate=0.1):
    for param in child.parameters():
        if random.random() < mutation_rate:
            param.data += torch.randn_like(param.data) * 0.1



def select_parents(population, fitness_scores, num_parents):
    parents = []
    for _ in range(num_parents):
        tournament = random.sample(list(zip(population, fitness_scores)), 4)  # Sample 5 networks
        tournament.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness
        parents.append(tournament[0][0])  # Select the best
    return parents


def evolve(population, num_generations, num_parents, mutation_rate=0.1):
    for generation in range(num_generations):
        print(f"New generation: {generation}")
        fitness_scores = [fitness_function(network) for network in population]

        parents = select_parents(population, fitness_scores, num_parents)

        next_generation = parents[:]
        #match initial pop
        while len(next_generation) < len(population):
            # Crossover
            parent_pair = random.sample(parents, 2)
            child = crossover(parent_pair)
            # Mutation
            mutate(child, mutation_rate)
            next_generation.append(child)
        population = next_generation
        
        best_fitness = max(fitness_scores)

    return population, best_fitness, fitness_scores



population_size = 50
num_generations = 100
num_parents = 20

population = initialise_population(population_size)
evolved_population, best_fitness, fitness_scores = evolve(population, num_generations, num_parents)

best_network_index = None
for i in range(len(fitness_scores)):
   if fitness_scores[i] == best_fitness:
       torch.save(evolved_population[i], "nn.pth")

##run game
