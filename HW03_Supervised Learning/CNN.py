import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        # (TODO) Design your CNN, it can only be less than 3 convolution layers
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 假設輸入圖片大小為 64x64
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # (TODO) Forward the model
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  

        x = x.view(x.size(0), -1) # 拉平成一維向量
        x = F.relu(self.fc1(x)) # 全連接層
        x = self.fc2(x)  

        return x

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device) -> float:
    # (TODO) Train the model and return the average loss of the data, we suggest use tqdm to know the progress
    model.train()  
    running_loss = 0.0
    total = 0
    correct = 0
    
    for data, target in tqdm(train_loader, desc="Training"):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()  
        output = model(data)  
        loss = criterion(output, target)  
        loss.backward()  
        optimizer.step()  
        
        running_loss += loss.item()  
        _, predicted = output.max(1)  
        total += target.size(0)  
        correct += predicted.eq(target).sum().item()  
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    # (TODO) Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
    model.eval() 
    running_loss = 0.0
    total = 0
    correct = 0
    
    with torch.no_grad():  
        for data, target in tqdm(val_loader, desc="Validating"):
            data, target = data.to(device), target.to(device)
            
            output = model(data) 
            loss = criterion(output, target)  
            
            running_loss += loss.item() 
            _, predicted = output.max(1) 
            total += target.size(0) 
            correct += predicted.eq(target).sum().item()  
    
    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def test(model: CNN, test_loader: DataLoader, criterion, device):
    # (TODO) Test the model on testing dataset and write the result to 'CNN.csv'
    model.eval()  
    predictions = []
    ground_truths = []
    
    with torch.no_grad():  
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            
            output = model(data)  
            _, predicted = output.max(1)  
            
            predictions.extend(predicted.cpu().numpy())  
            ground_truths.extend(target.cpu().numpy())  
    
    # 儲存預測結果為 CSV 檔案
    df = pd.DataFrame({"GroundTruth": ground_truths, "Prediction": predictions})
    df.to_csv("CNN.csv", index=False)
    print(f"Predictions saved to 'CNN.csv'")