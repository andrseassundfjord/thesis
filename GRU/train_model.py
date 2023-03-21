import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        hidden = torch.zeros(model.gru.num_layers, inputs.size(0), model.gru.hidden_size).to(device)
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        n_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
        n_total += inputs.size(0)
        
    avg_loss = total_loss / n_total
    accuracy = n_correct / n_total
    return avg_loss, accuracy

def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            hidden = torch.zeros(model.gru.num_layers, inputs.size(0), model.gru.hidden_size).to(device)
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            n_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            n_total += inputs.size(0)
        
    avg_loss = total_loss / n_total
    accuracy = n_correct / n_total
    return avg_loss, accuracy