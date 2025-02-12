import socket

HOST = "127.0.0.1"  # Listen on all interfaces
PORT = 5000       # Must match the port in Lua script

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print(f"Listening on {HOST}:{PORT}...")

conn, addr = server.accept()
print(f"Connected by {addr}")

while True:
    try:
        data = conn.recv(50240).decode("utf-8")
        if not data:
            pass
        print("Received from Lua:", data)
        
        # Send a response
        response = f"ACK: {data}"
        conn.sendall(response.encode("utf-8"))


    except KeyboardInterrupt:
        break
    except TimeoutError as e:
        print("Timeout error: ", e)
        break
    except Exception as e:
        print("Error:", e)
        break

conn.close()
server.close()
