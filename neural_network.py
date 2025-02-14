import neat
import neat.config
import neat.population
import subprocess
import os
import socket
from data_handler import DataHandler
import time

HOST = "127.0.0.1"
PORT = 5000

def stop_server(server_process):
    server_process.terminate()
    server_process.wait()

def start_emulator():
    # Launch the emulator using subprocess (adjust the path if needed)
    return subprocess.Popen([r".\Mesen.exe", "game\Super Mario Bros (E).nes", "data_parser.lua"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def stop_emulator(emulator_process):
    emulator_process.terminate()
    emulator_process.wait()

def eval_genome(genome, config):
    neural_net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = 0

    try:
        #start server
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((HOST,PORT))
            sock.settimeout(5)
            #start emulator
            emulator_process = start_emulator()
            #lmao
            time.sleep(2)

            packet_counter = 0
            #run network
            while True:
                data, addr = sock.recvfrom(10240)  # Receive game data
                
                packet_counter += 1
                if packet_counter <= 40:
                    continue  
                # Parse the data
                mario, tiles, enemies, score = DataHandler.parse_game_data(data.decode())
                
                flattened_data = DataHandler.flatten_game_data(mario, tiles, enemies)
                flattened_score = DataHandler.flatten_score(score)


                # Feed data to the neural network to decide actions
                output = neural_net.activate(flattened_data)


                # Convert the inputs dictionary to a message to send to the server
                action_str = ",".join([f"{value}" for value in output])

                sock.sendto(action_str.encode(), addr)
                fitness = fitness_function(flattened_score)
                # Check for game over condition, assuming the server sends it in the data
                if fitness < 0:
                    print("failed with fitness:", fitness)
                    break  # Exit the loop if game over

    except socket.timeout:
        fitness = -100  

    genome.fitness = fitness  

    stop_emulator(emulator_process)


previous_position = None
no_movement_timer = 0 

def fitness_function(score):
    global previous_position, no_movement_timer
    
    if previous_position is None:
        previous_position = score[0]
        
    #if only bigger by one or smaller by -1 (standing still)
    if score[0] >= previous_position + 1 or score[0] <= previous_position -1:
        no_movement_timer += 1  # Increment the no movement timer
    else:
        no_movement_timer = 0  # Reset timer if Mario moved
        
    if no_movement_timer > 120:
        return -500
    
    previous_position = score[0]
    
    #stop conditions
    if score[1] < 390 and score[0] <= 5: #didn't move for too long
        print(score[1])
        print(score[0])
        return -500
    if score[2] < 2:
        return -500

    score = (score[0] * 2) + score[1]
    
    return score



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        eval_genome(genome, config)  # Evaluate each genome using the eval_genome function

def run(config_path):
    # Start NEAT with the configuration file
    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT for a set number of generations (e.g., 50)
    winner = p.run(eval_genomes, 50)

if __name__ == "__main__":
    # Determine the local directory and config file path
    local_dir = os.path.dirname(__file__)  # Correcting to `os.path.dirname` instead of `os.path.dir_name`
    config_path = os.path.join(local_dir, "config.txt")

    # Start the run process with the provided config path
    run(config_path)
