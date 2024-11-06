import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import modify
import modify.kld as kld

torch.manual_seed(0)

in_features = 28*28
hidden_features = 100
out_features = 10

#model = modify.Sequential([
#    nn.Linear(in_features, hidden_features),
#    modify.ElementwiseNonlin(torch.relu, hidden_features),
#    nn.Linear(hidden_features, hidden_features),
#    modify.ElementwiseNonlin(torch.relu, hidden_features),
#    nn.Linear(hidden_features, hidden_features),
#    modify.ElementwiseNonlin(torch.relu, hidden_features),
#    nn.Linear(hidden_features, out_features),
#])
#model = modify.Sequential([
#    nn.Linear(in_features, out_features),
#])

model = nn.Linear(in_features, out_features)

#grad_model = kld.kldify(model, indep_across_layers=False)
#
## Data loading and preprocessing
#transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.1307,), (0.3081,))
#])
#
#train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
#test_dataset = datasets.MNIST('./data', train=False, transform=transform)
#
#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=1000)
#
## Initialize model, loss function and optimizer
#device = torch.device('mps')
#model = model.to(device)
#grad_model = grad_model.to(device)
#criterion = nn.CrossEntropyLoss()
#model_opt      = torch.optim.SGD(model.parameters(), lr=0.00001)
#grad_model_opt = torch.optim.Adam(grad_model.parameters(), lr=0.001)
#
## Training loop
#def train_model(epochs):
#    model.train()
#    for epoch in range(epochs):
#        for batch_idx, (image, target) in enumerate(train_loader):
#            image, target = image.view(-1, 28*28).to(device), target.to(device)
#
#            #model training based on real targets
#            output = model(image)
#            model_opt.zero_grad()
#            model_loss = criterion(output, target)
#            model_loss.backward()
#            kld.natural_grad(model, grad_model)
#            model_opt.step()
#            if batch_idx % 100 == 0:
#                print(f'Epoch: {epoch}, Batch: {batch_idx}, Model loss: {model_loss.item():.4f}')
#
#def train_grad_model(epochs):
#    model.train()
#    for epoch in range(epochs):
#        for batch_idx, (image, _) in enumerate(train_loader):
#            image = image.view(-1, 28*28).to(device)
#            #model_grad training based on targets sampled from the model.
#            output = model(image)
#            model_opt.zero_grad()
#            grad_model_opt.zero_grad()
#            sampled_target = torch.distributions.Categorical(logits=output).sample()
#            sampled_model_loss = criterion(output, sampled_target)
#            sampled_model_loss.backward()
#
#            grad_model_loss = kld.grad_model_loss(model, grad_model) 
#            grad_model_loss.backward()
#            grad_model_opt.step()
#            
#            if batch_idx % 100 == 0:
#                print(f'Epoch: {epoch}, Batch: {batch_idx}, Grad Model loss: {grad_model_loss.item():.4f}')
#                print(kld.rms_grad(model))
#
#def train(epochs):
#    model.train()
#    for epoch in range(epochs):
#        for batch_idx, (image, target) in enumerate(train_loader):
#            #image, target = image.view(-1, 28*28).to(device), target.to(device)
#            #model_opt.zero_grad()
#            #grad_model_opt.zero_grad()
#
#            #output = model(image)
#            #model_loss = criterion(output, target)
#            #model_loss.backward()
#
#            #grad_model_loss = kld.grad_model_loss(model, grad_model) 
#            #grad_model_loss.backward()
#
#            #grad_model_opt.step()
#            #model_opt.step()
#            #
#            #if batch_idx % 100 == 0:
#            #    print(f'Epoch: {epoch}, Batch: {batch_idx}, Model loss: {model_loss.item():.4f}, Grad Model loss: {grad_model_loss.item():.4f}')
#
#            image, target = image.view(-1, 28*28).to(device), target.to(device)
#
#            #model training based on real targets
#            output = model(image)
#            model_opt.zero_grad()
#            model_loss = criterion(output, target)
#            model_loss.backward()
#            kld.natural_grad(model, grad_model)
#            rms_grad_model = kld.rms_grad(model)
#            model_opt.step()
#            model_loss = model_loss.item()
#
#            #model_grad training based on targets sampled from the model.
#            output = model(image)
#            model_opt.zero_grad()
#            grad_model_opt.zero_grad()
#            sampled_target = torch.distributions.Categorical(logits=output).sample()
#            sampled_model_loss = criterion(output, sampled_target)
#            sampled_model_loss.backward()
#
#            grad_model_loss = kld.grad_model_loss(model, grad_model) 
#            grad_model_loss.backward()
#            grad_model_opt.step()
#            grad_model_loss = grad_model_loss.item()
#            
#            if batch_idx % 100 == 0:
#                print(rms_grad_model)
#                print(f'Epoch: {epoch}, Batch: {batch_idx}, Model loss: {model_loss:.4f}, Grad Model loss: {grad_model_loss:.4f}')
#
## Evaluation function
#def test():
#    model.eval()
#    test_loss = 0
#    correct = 0
#    with torch.no_grad():
#        for image, target in test_loader:
#            image, target = image.view(-1, 28*28).to(device), target.to(device)
#            output = model(image)
#            test_loss += criterion(output, target).item()
#            pred = output.argmax(dim=1, keepdim=True)
#            correct += pred.eq(target.view_as(pred)).sum().item()
#
#    test_loss /= len(test_loader)
#    accuracy = 100. * correct / len(test_loader.dataset)
#    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
#
## Run training
#if __name__ == '__main__':
#    #kld.sample_grad_model(model, grad_model)
#    train_grad_model(epochs=3)
#    #train_model(epochs=1)
#    train(epochs=5)
#    #test()
