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
    command = ["Mesen.exe", "game\Super Mario Bros (E).nes", "memory_reader.lua"]
    subprocess.run(command, check=True, capture_output=True, text=True)

def run_model(network):
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
    
    
    
model = NeuralNetwork()
model =torch.load("nn.pth", weights_only= False)
run_model(model)