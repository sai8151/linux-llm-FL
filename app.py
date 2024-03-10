import torch
import torch.nn as nn
import random
# Read the dataset and preprocess the text
with open('dataset.txt', 'r') as file:
    text = file.read().lower()

# Create a character-level vocabulary
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Define the RNN model
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

# Set the hyperparameters
input_size = len(chars)  # Number of unique characters in the dataset
output_size = len(chars)  # Number of unique characters in the dataset
hidden_size = 512  # Increase hidden size for potentially more complex data
num_layers = 4  # Increase number of layers for potentially more complex data
seq_length = 350  # Increase sequence length to capture longer contexts
learning_rate = 0.0005  # Slightly reduce learning rate for better stability
num_epochs = 2000  # Adjust based on the convergence of the model and validation performance


# Initialize the model, loss function, and optimizer
model = CharRNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    # Randomly sample a sequence from the text
    start_idx = random.randint(0, len(text) - seq_length - 1)
    end_idx = start_idx + seq_length + 1
    input_seq = text[start_idx:end_idx]
    target_seq = text[start_idx + 1:end_idx + 1]

    # Convert the input and target sequences to numerical representation
    input_tensor = torch.tensor([char_to_idx[ch] for ch in input_seq], dtype=torch.long).unsqueeze(0)
    target_tensor = torch.tensor([char_to_idx[ch] for ch in target_seq], dtype=torch.long)

    # Initialize hidden state
    hidden = torch.zeros(num_layers, 1, hidden_size)

    # Forward pass
    optimizer.zero_grad()
    output, _ = model(input_tensor, hidden)
    loss = criterion(output, target_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'language_model.pth')
