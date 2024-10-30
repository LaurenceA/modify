import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import modify
import modify.kld as kld

in_features = 28*28
hidden_features = 100
out_features = 10

model = modify.Sequential([
    nn.Linear(in_features, hidden_features),
    modify.ElementwiseNonlin(torch.relu, hidden_features),
    nn.Linear(hidden_features, hidden_features),
    modify.ElementwiseNonlin(torch.relu, hidden_features),
    nn.Linear(hidden_features, hidden_features),
    modify.ElementwiseNonlin(torch.relu, hidden_features),
    nn.Linear(hidden_features, out_features),
])

grad_model = kld.kldify(model)

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Initialize model, loss function and optimizer
device = torch.device('mps')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(epochs):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (image, target) in enumerate(train_loader):
            image, target = image.view(-1, 28*28).to(device), target.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

# Evaluation function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, target in test_loader:
            image, target = image.view(-1, 28*28).to(device), target.to(device)
            output = model(image)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Run training
if __name__ == '__main__':
    train(epochs=5)
    test()
