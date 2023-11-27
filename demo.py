import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10,5)
        self.l2 = nn.Linear(5,3)
        self.output = nn.Linear(3,1)
        self.sigmoid = nn.Sigmoid()
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
    def forward(self, x):
        x = self.act1(self.l1(x)) 
        x = self.act2(self.l2(x))  
        x = self.sigmoid(self.output)                                                             
model = mymodel()  
model.load_state_dict(torch.load('model.pt'))  
model.eval()
def predict_with_model():
    print("Input value. For Contract, 1 is month-to-month, 2 is one year, 3 is two years, 0 is not applicable. For others, 1 is yes, 0 is no, -1 is not applicable. See prompts ")
    features = ['Partner[0, 1]', 'SeniorCitizen[0, 1]]', 'Dependents[0, 1]]', 'OnlineSecurity[-1, 0, 1]', 'TechSupport[-1, 0, 1]', 
                'InternetService[0,1,2]', 'DeviceProtection[-1, 0, 1]', 'StreamingTV[-1, 0, 1]', 'StreamingMovies[-1, 0, 1]', 'Contract[0, 1, 2, 3]']
    
    user_input = []
    for feature in features:
        value = input(f"Enter value for {feature}: ")
        user_input.append(float(value))
    input_array = np.array(user_input, dtype=float).reshape(1, -1)
    input_tensor = torch.from_numpy(input_array).float()
    with torch.no_grad():
        output = model(input_tensor)
    probability = output.numpy()[0][0]
    return probability
predicted_probability = predict_with_model()
print(f"Predicted probability of non-returning: {predicted_probability}")