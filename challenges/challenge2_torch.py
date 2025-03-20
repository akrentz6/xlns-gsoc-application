import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import time

# Define the fully connected network as a subclass of nn.Module:
class FCNet(nn.Module):
    def __init__(self):

        super(FCNet, self).__init__()
        
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

        # Initialize: weights ~ N(0, 0.1) and biases = 0.
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):

        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))

        # We don't apply softmax here as the CrossEntropyLoss function
        # internally applies the appropriate log softmax transformation.
        x = self.fc2(x)

        return x

# Set up MNIST datasets with basic transforms (converting images to tensors)
train_transform = transforms.ToTensor()
raw_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
raw_test_dataset = datasets.MNIST('./data', train=False, download=True, transform=train_transform)

# Convert the datasets into in-memory tensors as they are loaded on the fly by default.
# Note: raw_train_dataset.data is of shape [60000, 28, 28] and is of type torch.uint8.
# We unsqueeze to add a channel dimension and convert to float, scaling to [0,1].
train_data = raw_train_dataset.data.unsqueeze(1).float() / 255.0
train_targets = raw_train_dataset.targets
test_data = raw_test_dataset.data.unsqueeze(1).float() / 255.0
test_targets = raw_test_dataset.targets

# Create TensorDatasets and DataLoaders based on the in-memory data.
train_dataset = TensorDataset(train_data, train_targets)
test_dataset = TensorDataset(test_data, test_targets)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCNet().to(device)

# Use the built-in cross entropy loss (fused log softmax + NLL).
loss_func = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.1)

start = time.time()
num_epochs = 7
for epoch in range(1, num_epochs + 1):
    
    # Training phase
    model.train()
    
    # Track cumulative loss and accuracy for the training epoch
    running_train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for data, target in train_loader:
        
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad()

        outputs = model(data)
        loss = loss_func(outputs, target)
        
        loss.backward()
        optimiser.step()
        
        # Accumulate the loss (multiplied by the batch size for averaging later)
        running_train_loss += loss.item() * data.size(0)

        _, predicted = torch.max(outputs, dim=1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()
    
    # Calculate the average training loss and accuracy for the epoch
    train_epoch_loss = running_train_loss / train_total
    train_epoch_acc = train_correct / train_total

    # Validation phase
    model.eval()
    
    # Track cumulative loss and accuracy for the validation epoch
    running_val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    # Disable gradient calculation for validation
    with torch.no_grad():
        for data, target in test_loader:
            
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = loss_func(outputs, target)
            
            # Accumulate the loss (multiplied by the batch size for averaging later)
            running_val_loss += loss.item() * data.size(0)

            _, predicted = torch.max(outputs, 1)
            
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()
    
    # Calculate the average validation loss and accuracy for the epoch
    val_epoch_loss = running_val_loss / val_total
    val_epoch_acc = val_correct / val_total

    # Print epoch summary
    print(f"Epoch {epoch}:")
    print(f"  Training   - Loss = {train_epoch_loss:.4f}, Accuracy = {train_epoch_acc:.4f}")
    print(f"  Validation - Loss = {val_epoch_loss:.4f}, Accuracy = {val_epoch_acc:.4f}")

elapsed = time.time() - start

# Final test accuracy (optional, as we've already been validating each epoch)
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        
        data, target = data.to(device), target.to(device)

        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        
        total += target.size(0)
        correct += (predicted == target).sum().item()

print("\nFinal Test Accuracy:", correct / total)
print("Elapsed time:", elapsed)