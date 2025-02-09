import mmap
import os
import time

MESEN_PATH = os.path.join(os.getenv("USERPROFILE"), "Documents", "Mesen2")
SHARED_MEMORY_FILE = os.path.join(MESEN_PATH, "data")

def parse_value(value):
    """Convert string value into int or list of ints if formatted as {1, 2, 3}."""
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):  # Detect list format
        return [int(v.strip()) for v in value[1:-1].split(",") if v.strip()]
    return int(value)  # Otherwise, return as integer

score_key = [
    "timer", "player_x_screen", "level", "world", "player_lives"
]

class emu_read():
    def read_game_memory():
        try:
            with open(SHARED_MEMORY_FILE, "r+b") as f:
                if os.path.getsize(SHARED_MEMORY_FILE) == 0:
                    return {}, {}  # Return empty if file is empty

                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
                mm.seek(0)
                data = mm.read().decode().strip()

                if not data:
                    mm.close()
                    return {}, {}

                game_data = {}
                score = {}

                for line in data.split("\n"):
                    try:
                        key, value = line.split(":")
                        if key in score_key:
                            score[key] = parse_value(value)
                        else:
                            game_data[key] = parse_value(value)
                    except ValueError:
                        print(f"Skipping invalid line: {line}")

                mm.close()
                return game_data, score

        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"Error reading memory file: {e}")
            return {}, {}
                


    def normalise_data(data):
        normalised_data = []
        for key,value in data.items():

            if isinstance(value, list):
                for v in value:
                    normalised_data.append(v/255)
            else:
                normalised_data.append(value/255)

        return normalised_data
    
    