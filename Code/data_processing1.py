import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim

# Load the data and labels
path = os.path.abspath(".")
features = torch.load(path + "/Data/cdr3_features.pt")
labels_mhca = torch.load(path + "/Data/mhca_labels.pt")

dataset = TensorDataset(features, labels_mhca)
dataloader = DataLoader(dataset=dataset, batch_size=70000, shuffle=True)

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(features, labels_mhca, test_size=0.2, random_state=42)

#Define the Model
class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output = self.fc1(input)
        output = self.relu(output)
        output = self.fc2(output)

        return output
    
input_size = 38 * 20
hidden_size = 500
output_size = 119

model = SimpleNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
criterian = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.1)

num_epoch = 0
for epoch in range(num_epoch):
    for x, y in dataloader:
        optimizer.zero_grad()
        input = x.view(-1, 38*20)
        output = model.forward(input)
        loss = criterian(output, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch{epoch+1}: loss = {loss}")

torch.save(model, path + "/Model/model_state_dict.pth")

model.eval()

with torch.no_grad():
    inputs = X_test.view(-1, 38*20)
    outputs = model(inputs)
    # print(outputs)
    a, predicted = torch.max(outputs.data, 1)
    total = y_test.size(0)
    print(predicted)
    unique_classes = torch.unique(predicted)
    num_unique_classes = len(unique_classes)
    print("hhhhhh:", num_unique_classes)
    correct = (predicted == y_test).sum().item()

    accuracy = correct / total
    print(f'Accuracy of the network on the test samples: {accuracy * 100}%')
    # print(a)
