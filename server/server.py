import socket
import pickle
import torch
import torch.nn as nn
# import s2
# Define global variables
server_ip = '0.0.0.0'        # Server IP address
server_port = 12345          # Server port
learning_rate = 0.001        # Learning rate for updating model parameters
num_clients = 0              # Number of clients participating in federated learning
import requests

# API endpoints
BASE_URL = "https://your_domain.com/api/"
UPLOAD_URL = BASE_URL + "upload.php"
DOWNLOAD_URL = BASE_URL + "download.php"
DELETE_URL = BASE_URL + "delete.php"

# Authentication token
AUTH_KEY = "iith"

# Function to upload model file
def upload_model(file_path):
    headers = {'Authorization': f'Bearer {AUTH_KEY}'}
    files = {'file': open(file_path, 'rb')}
    response = requests.post(UPLOAD_URL, files=files, headers=headers)
    print(response.text)

# Function to download model file
def download_model():
    headers = {'Authorization': f'Bearer {AUTH_KEY}'}
    response = requests.get(DOWNLOAD_URL, headers=headers)
    if response.status_code == 200:
        with open('downloaded_model.llm', 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    else:
        print("Failed to download model.")

# Function to delete model file
def delete_model():
    headers = {'Authorization': f'Bearer {AUTH_KEY}'}
    response = requests.delete(DELETE_URL, headers=headers)
    print(response.text)




# Define the CharRNN model
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        return out, hidden
def send_file(file_path, client_ip, client_port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.connect((client_ip, client_port))
    with open(file_path, 'rb') as file:
        data = file.read(1024)
        while data:
            server_socket.send(data)
            data = file.read(1024)
    server_socket.close()

# Load the dataset and preprocess the text
with open('dataset.txt', 'r') as file:
    text = file.read().lower()

# Create a character-level vocabulary
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Load the trained model
input_size = len(chars)  # Update with your input size
output_size = len(chars)  # Update with your output size
hidden_size = 512
num_layers = 4


model = CharRNN(input_size, hidden_size, output_size, num_layers)
model.load_state_dict(torch.load('language_model.pth'))
model.eval()

# Create socket and bind to server address
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(5)  # Listen for up to 5 client connections
# s2.send_file('server_model.pth', '127.0.0.2', 12346)

print("Waiting for client connections...")

try:
    aggregated_gradients = None  # Initialize aggregated gradients
    while True:
        # Accept connection from client
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with client {client_address}")

        # Receive serialized gradients from client
        serialized_gradients = b''
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            serialized_gradients += data

        # Deserialize gradients
        gradients = pickle.loads(serialized_gradients)

        # Aggregate gradients
        if not aggregated_gradients:
            aggregated_gradients = gradients
        else:
            aggregated_gradients = [ag + g for ag, g in zip(aggregated_gradients, gradients)]
        
        num_clients += 1  # Increment the number of connected clients

        print(f"Gradients received from client {client_address}")

        # Close connection with client
        client_socket.close()

        # Update global model parameters using aggregated gradients
        with torch.no_grad():
            for param, grad_sum in zip(model.parameters(), aggregated_gradients):
                if grad_sum is not None:
                    param -= learning_rate * grad_sum / num_clients  # Update model parameters

        # Save the updated model
        torch.save(model.state_dict(), 'server_model.pth')
        delete_model()
        upload_model("server_model.pth")
        #download_model()
        
        # print("Model updated and saved.")
        # send_file('server_model.pth', '127.0.0.1', 12346)

except KeyboardInterrupt:
    print("\nServer terminated by user.")
finally:
    # Close server socket
    server_socket.close()



