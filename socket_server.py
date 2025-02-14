import socket
from data_handler import DataHandler

HOST = "127.0.0.1"
PORT = 5000

# Create a UDP socket
udp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_server.bind((HOST, PORT))

while True:
    try:
        data, addr = udp_server.recvfrom(10240)  # Receive up to 1024 bytes
        mario, tiles, enemies, score = DataHandler.parse_game_data(data.decode())
        flattened_data = DataHandler.flatten_game_data(mario,tiles,enemies)
        flattened_score = DataHandler.flatten_score(score)
        
        response = f"Message received: {data.decode()}"
        udp_server.sendto(response.encode(), addr)
    except KeyboardInterrupt:
        print('Server end')
        break

