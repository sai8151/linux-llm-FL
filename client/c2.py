import socket
import pickle

def send_file(file_path, client_ip, client_port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.connect((client_ip, client_port))
    with open(file_path, 'rb') as file:
        data = file.read(1024)
        while data:
            server_socket.send(data)
            data = file.read(1024)
    server_socket.close()

def receive_file(file_path, server_port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.2', server_port))
    server_socket.listen(1)
    client_socket, client_address = server_socket.accept()
    with open(file_path, 'wb') as file:
        data = client_socket.recv(1024)
        while data:
            file.write(data)
            data = client_socket.recv(1024)
    client_socket.close()
    server_socket.close()

# Example usage
# send_file('server_model.pth', '127.0.0.1', 12345)
receive_file('client_model.pth', 12346)
