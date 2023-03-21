import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from split_data import split_data
from gru_model import GRUModel

from split_dataset import get_dataloaders

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the hyperparameters
input_dim = 20
hidden_dim = 32
output_dim = 2
num_layers = 1
dropout_prob = 0.2
learning_rate = 0.001
num_epochs = 10

# Load the data
train_loader, test_loader = get_dataloaders()

# Define the model
model = GRUModel(input_dim, hidden_dim, output_dim, num_layers, dropout_prob)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # Train the model for one epoch
    train_loss = 0.0
    correct = 0
    total = 0
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss /= len(train_loader.dataset)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Evaluate the model on the test set
    test_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    # Print the training and testing losses and accuracies
    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# Plot the training and testing losses and accuracies
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(train_losses, label='Training')
axs[0].plot(test_losses, label='Testing')
axs[0].set_title('Loss over Epochs')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[1].plot(train_accuracies, label='Training')
axs[1].plot(test_accuracies, label='Testing')
axs[1].set_title('Accuracy over Epochs')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy (%)')
axs[1].legend()
plt.show()
