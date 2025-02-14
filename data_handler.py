import re
import numpy as np

class DataHandler:
    @staticmethod
    def parse_game_data(data):
        # Extract Mario data
        mario_match = re.search(r'Mario: \[(.*?)\]', data)
        mario = []
        if mario_match:
            mario_data = mario_match.group(1)
            mario = [{k: float(v) if '.' in v else int(v) for k, v in re.findall(r'(\w+)=([-+]?\d*\.?\d+)', mario_data)}]

        # Extract Tile data
        tiles_match = re.search(r'Tile: \[(.*?)\]', data, re.DOTALL)
        tiles = []
        if tiles_match:
            tiles_data = tiles_match.group(1)
            tiles = [{k: int(v)} for k, v in re.findall(r'(\d+_\d+)=(\d+)', tiles_data)]

        # Extract Enemy data
        enemies_match = re.search(r'Enemies: \[(.*?)\]', data, re.DOTALL)
        enemies = []
        if enemies_match:
            enemies_data = enemies_match.group(1)
            enemy_list = re.findall(r'{(.*?)}', enemies_data)
            for enemy in enemy_list:
                if "nil" not in enemy:  # Ignore empty enemies
                    enemies.append({k: (float(v) if '.' in v else int(v)) if v != "nil" else None 
                                    for k, v in re.findall(r'(\w+)=([-+]?\d*\.?\d+|nil)', enemy)})
                    
                    
        score_match = re.search(r'Score: \{(.*?)\}', data)
        score = {}
        if score_match:
            score_data = score_match.group(1)
            score = {k: float(v) if '.' in v else int(v) for k, v in re.findall(r'(\w+)=([-+]?\d*\.?\d+)', score_data)}
            
            
            
        return mario, tiles, enemies, score
    

    @staticmethod
    def flatten_game_data(mario, tiles, enemies):
        # Flatten Mario Data (x, y, vx, vy in order)
        mario_flat = [mario[0]['x'], mario[0]['y'], mario[0]['vx'], mario[0]['vy']]

        # Sort tiles by position (to maintain order)
        tile_flat = [v for k, v in sorted([(list(tile.keys())[0], list(tile.values())[0]) for tile in tiles])]

        # Flatten Enemies (x, y, vx, vy, t in order for each enemy)
        enemies_flat = []
        for enemy in enemies:
            enemies_flat.extend([enemy['x'], enemy['y'], enemy['vx'], enemy['vy'], enemy['t']])

        # Combine into one 1D array
        return np.array(mario_flat + tile_flat + enemies_flat, dtype=np.float32)
    
    
    @staticmethod
    def flatten_score(score):
        score_flat = [score['d'], score['t'], score['l']]
        return np.array(score_flat, dtype=np.float32)


    

