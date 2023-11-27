import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10,30)
        self.l2 = nn.Linear(30,30)
        self.l3 = nn.Linear(30,10)
        self.output = nn.Linear(10,1)
        self.sigmoid = nn.Sigmoid()
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
    def forward(self, x):
        x = self.act1(self.l1(x)) 
        x = self.act2(self.l2(x))  
        x = self.act3(self.l3(x)) 
        x = self.sigmoid(self.output(x))  
        return x
model = mymodel()  
model.load_state_dict(torch.load('model.pt'))  
model.eval()
data = pd.read_csv('test_data.csv')
data['TotalCharges'].apply(float)
datam = data[data['gender']==-1]
dataf = data[data['gender']==1]
datam = datam[['Partner','SeniorCitizen','Dependents','OnlineSecurity','TechSupport','InternetService','DeviceProtection','StreamingTV','StreamingMovies','Contract','Churn']]
dataf = dataf[['Partner','SeniorCitizen','Dependents','OnlineSecurity','TechSupport','InternetService','DeviceProtection','StreamingTV','StreamingMovies','Contract','Churn']]
x = datam.iloc[:,:-1].to_numpy()
y = datam.iloc[:, -1].to_numpy() 
x1 = dataf.iloc[:,:-1].to_numpy()
y1 = dataf.iloc[:, -1].to_numpy()
X_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()
X_tensor1 = torch.from_numpy(x1).float()
y_tensor1 = torch.from_numpy(y1).float()
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy of the model on the test data upon female: {accuracy}')
dataset = TensorDataset(X_tensor1, y_tensor1)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy of the model on the test data upon male: {accuracy}')

