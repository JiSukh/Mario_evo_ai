from neural import NeuralNetwork, fitness_function
import torch
import subprocess
import threading
import emu_data_reader as er
import time

model = NeuralNetwork()
model.load_state_dict(torch.load("nn.pth"))
fitness_function(model)


def run_game():
    # Run Mesen as a subprocess in a separate thread
    command = ["Mesen.exe", "game\Super Mario Bros (E).nes", "memory_reader.lua"]
    subprocess.run(command, check=True, capture_output=True, text=True)

def fitness_function(network):
    game_thread = threading.Thread(target=run_game)
    game_thread.start()
    
    while game_thread.is_alive():
        game_data,score = er.read_game_memory()
        if game_data:
            normalised_data = er.normalise_data(game_data)
            in_tensor = torch.FloatTensor(normalised_data)
            outputs = network(in_tensor)
            er.write_inputs(outputs)
        time.sleep(0.01)
    