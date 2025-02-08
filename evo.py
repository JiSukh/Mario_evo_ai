import mmap
import time

SHARED_MEMORY_FILE = "C:\\Users\\jsukh\\Documents\\Mario_evo_ai\\data"

def parse_value(value):
    """Convert string value into int or list of ints if formatted as {1, 2, 3}."""
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):  # Detect list format
        return [int(v.strip()) for v in value[1:-1].split(",") if v.strip()]
    return int(value)  # Otherwise, return as integer

def read_game_memory():
    with open(SHARED_MEMORY_FILE, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mm.seek(0)
        data = mm.read().decode().strip()
        if data:
            game_data = {}
            #str to dict
            for line in data.split("\n"):
                key, value = line.split(":")
                game_data[key] = parse_value(value)  # Convert values properly
            return game_data


data = read_game_memory()
print(data)
