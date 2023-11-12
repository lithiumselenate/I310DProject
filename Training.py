import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
data = pd.read_csv('train_data.csv')
data['TotalCharges'].apply(float)
x = data.iloc[:,:-1].to_numpy()
y = data.iloc[:, -1].to_numpy()
print(x.size)
print(y.size)
X_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# Convert to PyTorch tensors
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(19,57)
        self.l2 = nn.Linear(57,57)
        self.l3 = nn.Linear(57,2)
        self.output = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
    def forward(self, x):
        x = self.act1(self.l1(x)) 
        x = self.act2(self.l2(x))  
        x = self.act3(self.l3(x)) 
        x = self.sigmoid(self.output(x))  # Apply output to x, then apply sigmoid
        return x
nnmodel = model()
device = torch.device("cuda")
criterion = nn.BCELoss()
optimizer = optim.Adam(nnmodel.parameters(), lr=0.001)
num_epochs = 300  # Set the number of epochs
losses = []
accs = []
for epoch in range(num_epochs):
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    for inputs, labels in dataloader:
        outputs = nnmodel(inputs)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted = (outputs > 0.5).float()
        correct_predictions += (predicted == labels).float().sum().item()
        total_samples += labels.size(0)
    avg_loss = total_loss / total_samples
    avg_accuracy = correct_predictions / total_samples
    losses.append(avg_loss)
    accs.append(avg_accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.3f}, Accuracy: {avg_accuracy:.3f}")
torch.save(nnmodel.state_dict(), 'model.pt')
from matplotlib import pyplot as plt
plt.plot(losses)
plt.plot(accs,)
plt.xlabel("Epochs")
plt.show()

