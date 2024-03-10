import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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

# Load the dataset and preprocess the text
with open('dataset.txt', 'r') as file:
    text = file.read().lower()

# Create a character-level vocabulary
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Define a function to generate text continuation based on the trained model
def generate_text(model, start_text, num_chars=100, temperature=1.0):
    with torch.no_grad():
        # Initialize the hidden state
        hidden = None

        # Convert start_text to numerical representation
        input_tensor = torch.tensor([char_to_idx[ch] for ch in start_text], dtype=torch.long).unsqueeze(0)

        # Generate text continuation
        generated_text = start_text
        for _ in range(num_chars):
            output, hidden = model(input_tensor, hidden)
            # Use the temperature parameter to control the randomness of the output
            output_dist = output.squeeze().div(temperature).exp()
            selected_char_idx = torch.multinomial(output_dist, 1)[0]
            # Convert the selected character index back to the character
            selected_char = idx_to_char[selected_char_idx.item()]
            generated_text += selected_char
            # Update the input tensor with the latest character
            input_tensor = torch.tensor([selected_char_idx], dtype=torch.long).unsqueeze(0)
        
        return generated_text

# Load the trained model
input_size = len(chars)
output_size = len(chars)
hidden_size = 512
num_layers = 4

model = CharRNN(input_size, hidden_size, output_size, num_layers)
model.load_state_dict(torch.load('language_model.pth'))
model.eval()

# Define the training process with hypothetical loss values
losses = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.04, 0.03]

# Plot the loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses, color='blue')
plt.title('Loss Curve During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_curve.png')
plt.close()

# Plot the histogram of model weights
weights = model.fc.weight.detach().numpy().flatten()
plt.figure(figsize=(10, 5))
plt.hist(weights, bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of Model Weights')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('weights_histogram.png')
plt.close()

# Plot the gradients during training (hypothetical example)
num_iterations = 100
gradients = [np.random.uniform(-1, 1) for _ in range(num_iterations)]
plt.figure(figsize=(10, 5))
plt.plot(gradients, color='orange')
plt.title('Gradients During Training')
plt.xlabel('Iteration')
plt.ylabel('Gradient Value')
plt.grid(True)
plt.savefig('gradients_plot.png')
plt.close()

# Prompt the user for input
start_text = input("Enter the starting text for text generation: ")

# Generate text continuation
generated_text = generate_text(model, start_text, num_chars=200, temperature=0.80)

# Save generated text to a file
with open('generated_text.txt', 'w') as file:
    file.write(generated_text)

# Print generated text
print("Generated Text:")
print(generated_text)
