import csv
import torch
from torch import nn
from typing import Tuple
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader

class KeyPointDataset(Dataset):
    def __init__(self, csv_path: str):
        super().__init__()
        self.x = []
        self.y = []

        with open(csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)

            for row in csv_reader:
                tensor_row = torch.tensor([float(value) for value in row[:-1]])
                self.x.append(tensor_row)
                self.y.append(torch.tensor(float(row[-1])))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

class PoseNet(nn.Module):
    """
    使用姿态特征信息预测[0, 1]之间的得分。
    """
    def __init__(self, input_dim: int=17*2, hidden_dim: int=256):
        """
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x