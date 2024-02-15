import torch
from data_preprocessing import one_hot_encoding
import torch.nn as nn

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
model = torch.load("/Users/yuqiao/Desktop/MyFile/England/TB2/Data Science Mini Project/Code/dsmp-2024-group10/Model/model_state_dict.pth")
listmhcb = ['B2M', 'H-2Aa', 'H-2Eb1', 'HLA-DPB*04:01', 'HLA-DPB1*02:01',
    'HLA-DPB1*02:01:02', 'HLA-DPB1*03:01', 'HLA-DPB1*04:01', 'HLA-DPB1*13:01',
    'HLA-DQB1*02', 'HLA-DQB1*02:01', 'HLA-DQB1*02:01:08', 'HLA-DQB1*02:02:01:01',
    'HLA-DQB1*03:01', 'HLA-DQB1*03:02', 'HLA-DQB1*03:02:12', 'HLA-DQB1*05:01',
    'HLA-DQB1*05:01:01:03', 'HLA-DQB1*05:02', 'HLA-DQB1*06:01', 'HLA-DQB1*06:02',
    'HLA-DRA*01:01', 'HLA-DRB1*01', 'HLA-DRB1*01:01', 'HLA-DRB1*01:01:01',
    'HLA-DRB1*03:01', 'HLA-DRB1*04:01', 'HLA-DRB1*04:01:01', 'HLA-DRB1*07:01',
    'HLA-DRB1*09:01', 'HLA-DRB1*11:01', 'HLA-DRB1*11:01:02', 'HLA-DRB1*13:01',
    'HLA-DRB1*15', 'HLA-DRB1*15:01', 'HLA-DRB1*15:01:01:04', 'HLA-DRB1*15:02',
    'HLA-DRB1*15:02:02', 'HLA-DRB1*15:03', 'HLA-DRB3*02:02', 'HLA-DRB3*03:01',
    'HLA-DRB3*03:01:01', 'HLA-DRB5*01', 'HLA-DRB5*01:01', 'HLA-DRB5*01:01:01']
while True:
    print("Please input:")
    seq = str(input())
    seq_input = one_hot_encoding(seq=seq, max_len=38)
    seq_input = torch.tensor(seq_input, dtype=torch.float).view(38*20)
    output = model(seq_input)
    print(listmhcb[int(torch.argmax(output))])