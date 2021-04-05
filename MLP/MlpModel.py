import torch
import torch.nn as nn
import numpy as np
import pickle

import MLP.config as config

class MlpModel(nn.Module):
    def __init__(self, **kwargs):
        super(MlpModel, self).__init__()
        self.relu = nn.ReLU()

        self.hidden_size = 32
        self.fc1 = nn.Linear(config.INPUT_SIZE, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, config.OUTPUT_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x