import numpy as np
import torch
import torch.nn as nn
from emu_data_reader import emu_read as er
import random
import subprocess


    


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(3, 64)
        self.layer2 = nn.Linear(64, 13)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x


        
def initialise_population(size):
    population = []
    for i in size:
        network = NeuralNetwork()
        #randomise
        for param in network.parameters():
            param.data = torch.randn_like(param.data)

        population.append(network)

    return population

def fitness_function(network):
    

    timer_w = 1
    world_w = 5
    level_w = 3
    x_pos_w = 1
    lives_w = 10

    fitness = (score[0] * timer_w) + (score[1] * world_w) + (score[2] * level_w) + (score[3] * x_pos_w) * (score[4] * lives_w)

    fitness = fitness / 4
    return fitness

def crossover(parents):
    parent1, parent2 = parents
    child = NeuralNetwork()
    for param1, param2, child_param in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
        child_param.data = (param1.data + param2.data) / 2
    return child


def mutate(child, mutation_rate=0.01):
    for param in child.parameters():
        if random.random() < mutation_rate:
            param.data += torch.randn_like(param.data) * 0.1

def select_parents(population, fitness_scores, num_parents):
    parents = random.choices(population, weights=fitness_scores, k=num_parents)
    return parents


def evolve(population, num_generations, num_parents, mutation_rate=0.01):
    for generation in range(num_generations):
        # Evaluate fitness
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
        # Optionally, print the best fitness score of the generation
        best_fitness = max(fitness_scores)

    return population



population_size = 50
num_generations = 100
num_parents = 10

population = initialise_population(population_size)
evolved_population = evolve(population, num_generations, num_parents)