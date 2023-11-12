import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
data = pd.read_csv('train_data.csv')
data['TotalCharges'].apply(float)
x = data.iloc[:-1].to_numpy()
y = data.iloc[-1].to_numpy()
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
        self.output = nn.Linear(57,1)
        self.sigmoid = nn.Sigmoid()
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
    def forward(self, x):
        x = self.act1(self.l1)
        x = self.act2(self.l2)
        x = self.sigmoid(self.output(x))
        return x
nnmodel = model()
device = torch.device("cuda")
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10  # Set the number of epochs
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    outputs = (outputs>0.5).float()
    acc = (outputs==labels).floag().sum()
    print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,num_epochs, loss.data[0], acc/x.shape[0]))
torch.save(nnmodel.stae_dict(), 'model.pt')
