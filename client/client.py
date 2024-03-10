import streamlit as st
import torch
import torch.nn as nn
import socket
import pickle
from torchviz import make_dot
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
import requests
# import c2
# Define the CharRNN model
# API endpoints
BASE_URL = "https://your_domain.com/api/"
UPLOAD_URL = BASE_URL + "upload.php"
DOWNLOAD_URL = BASE_URL + "download.php"
DELETE_URL = BASE_URL + "delete.php"
DOWNLOAD_URL = "https://your_domain.com/api/uploads/server_model.pth"

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
        with open('downloaded_model.pth', 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    else:
        print(f"Failed to download model. Status code: {response.status_code}")

# Function to delete model file
def delete_model():
    headers = {'Authorization': f'Bearer {AUTH_KEY}'}
    response = requests.delete(DELETE_URL, headers=headers)
    print(response.text)


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


# def receive_file(file_path, server_port):
#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_socket.bind(('0.0.0.0', server_port))
#     server_socket.listen(1)
#     client_socket, client_address = server_socket.accept()
#     with open(file_path, 'wb') as file:
#         data = client_socket.recv(1024)
#         while data:
#             file.write(data)
#             data = client_socket.recv(1024)
#     client_socket.close()
#     server_socket.close()


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

# Define the text generation function
def generate_text(model, start_text, num_chars=100, temperature=1.0):
    with torch.no_grad():
        hidden = None
        input_tensor = torch.tensor([char_to_idx[ch] for ch in start_text], dtype=torch.long).unsqueeze(0)
        generated_text = start_text
        for _ in range(num_chars):
            output, hidden = model(input_tensor, hidden)
            output_dist = output.squeeze().div(temperature).exp()
            selected_char_idx = torch.multinomial(output_dist, 1)[0]
            selected_char = idx_to_char[selected_char_idx.item()]
            generated_text += selected_char
            input_tensor = torch.tensor([selected_char_idx], dtype=torch.long).unsqueeze(0)
        
        return generated_text

def train_model_with_generated_text(model, generated_text):
    # Preprocess the generated text
    generated_text_numerical = [char_to_idx[ch] for ch in generated_text]
    # Convert the generated text to a PyTorch tensor
    input_tensor = torch.tensor(generated_text_numerical[:-1], dtype=torch.long).unsqueeze(0)  # Remove last character
    target_tensor = torch.tensor(generated_text_numerical[1:], dtype=torch.long).unsqueeze(0)  # Shift by one for target

    # Forward pass to generate hidden state
    hidden = None
    output, _ = model(input_tensor, hidden)

    # Compute the loss using the generated text as target
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output.view(-1, output_size), target_tensor.view(-1))
    # Backward pass and update model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    gradients = [param.grad for param in model.parameters() if param.grad is not None]
    serialized_gradients = pickle.dumps(gradients)
    # Send gradients to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_port = 12345
    client_socket.connect(('localhost', server_port))
    client_socket.sendall(serialized_gradients)
    # Plot gradients
    fig1 = plt.figure() 
    plt.plot(gradients[0]) 
    st.pyplot(fig1)
    weights = model.fc.weight.detach().numpy().flatten()
    # Plot the histogram
    plt.figure(figsize=(10, 5))
    plt.hist(weights, bins=50)
    plt.title('Weight Histogram')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    # Display the plot in Streamlit
    st.pyplot(plt)
    # Close client socket
    client_socket.close()
    # Save the updated model parameters
    torch.save(model.state_dict(), 'language_model_finetuned.pth')


# Streamlit UI
st.title("Text Generation App")

start_text = st.text_input("Enter the starting text for text generation:")
num_chars = st.slider("Number of characters to generate:", 50, 500, 100)
temperature = st.slider("Temperature (controls randomness):", 0.1, 2.0, 1.0)
st.text("Generated Text:")


# c2.receive_file('client_model.pth', 12346)

if st.button("Generate Text"):
    generated_text = generate_text(model, start_text, num_chars, temperature)
    st.text("Generated Text:")
    st.write(generated_text)
    #st.button("Train Model with Generated Text")
    train_model_with_generated_text(model, generated_text)
    st.text("Model trained with generated text.")

    #st.stop()  # Stop Streamlit app after training model
# if st.button("get new model"):
#     receive_file('client_model.pth', 12346)
if st.button("pull new version"):
        # delete_model()
        # upload_model("server_model.pth")
        download_model()
