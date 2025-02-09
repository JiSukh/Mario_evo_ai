import mmap
import os
import mmap
import torch
from collections import OrderedDict

MESEN_PATH = os.path.join(os.getenv("USERPROFILE"), "Documents", "Mesen2")
SHARED_MEMORY_FILE = os.path.join(MESEN_PATH, "data")
SHARED_MEMORY_FILE_INPUT = os.path.join(MESEN_PATH, "input")




class emu_read():
    def read_game_memory():
        # Define the correct key order
        correct_order = [
            "enemies_x3", "enemies_y1", "Block_2_13", "Block_3_9", "player_x_level", "Block_1_10",
            "enemies_x0", "Block_7_11", "enemies_y3", "Block_2_14", "Block_1_11", "Block_6_9",
            "player_y_screen", "timer", "Block_2_8", "Block_5_14", "enemies_y4", "Block_5_10",
            "Block_5_11", "Block_4_14", "Block_2_11", "Block_1_14", "player_lives", "Block_1_12",
            "enemy_states4", "Block_6_12", "enemy_states0", "enemies_x1", "Block_6_10", "Block_4_13",
            "layout_index", "Block_3_14", "player_x_screen", "Block_2_9", "Block_5_8", "enemies_x4",
            "Block_1_8", "Block_2_12", "enemy_states3", "Block_7_13", "Block_7_9", "enemies_x2",
            "player_state", "Block_2_10", "Block_6_11", "Block_7_8", "Block_3_12", "Block_5_9",
            "Block_1_9", "player_y_level", "Block_6_8", "power_up_state", "player_vertical_speed",
            "Block_4_10", "Block_3_11", "Block_3_8", "Block_4_12", "enemy_states2", "Block_4_11",
            "Block_7_14", "Block_3_10", "Block_5_13", "Block_6_13", "Block_7_12", "Block_4_8",
            "Block_5_12", "Block_6_14", "world", "enemies_y2", "enemies_y0", "level", "player_horizontal_speed",
            "Block_3_13", "player_vertical_velocity", "player_float_state", "Block_1_13", "Block_7_10",
            "x", "Block_4_9", "enemy_states1"
        ]
        # Define the score keys
        score_key = [
            "timer", "player_x_level", "level", "world", "player_lives"
        ]
        
        try:
            with open(SHARED_MEMORY_FILE, "r+b") as f:
                if os.path.getsize(SHARED_MEMORY_FILE) == 0:
                    return OrderedDict(), OrderedDict()  # Return empty if file is empty

                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
                mm.seek(0)
                data = mm.read().decode().strip()

                if not data:
                    mm.close()
                    return OrderedDict(), OrderedDict()  # Return empty if no data

                parsed_data = OrderedDict()

                # Parse the data, assuming 'key:value' format
                for line in data.splitlines():
                    if ':' in line:
                        key, value = line.split(":", 1)
                        parsed_data[key.strip()] = int(value.strip())

                mm.close()

                # Separate out the score data based on score_key
                score_data = OrderedDict()
                for key in score_key:
                    if key in parsed_data:
                        score_data[key] = parsed_data[key]
                    else:
                        score_data[key] = 0  

                remaining_data = OrderedDict()
                for key in correct_order:
                    if key not in score_key:  
                        if key in parsed_data:
                            remaining_data[key] = parsed_data[key]
                        else:
                            remaining_data[key] = 0  

                return remaining_data, score_data  

        except (FileNotFoundError, ValueError, OSError) as e:
            return OrderedDict(), OrderedDict()  # Return empty on error
                


    def normalise_data(data):
        normalised_data = []
        for key,value in data.items():

            if isinstance(value, list):
                for v in value:
                    normalised_data.append(v/255)
            else:
                normalised_data.append(value/255)

        return normalised_data
    
    def write_inputs(inputs):
        inputs = inputs.tolist()
    
        with open(SHARED_MEMORY_FILE_INPUT, "w") as f:
            for i, value in enumerate(inputs):
                # Write 1 if value is greater than 0.7, otherwise write 0
                if value > 0.5:
                    f.write("1\n")
                else:
                    f.write("0\n")
        