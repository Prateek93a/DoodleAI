import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from helpers import load_categories

categories = load_categories('./categories.txt')


class NeuralNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.dropout1 = nn.Dropout2d(0.4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.4)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(64*4*4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 133)

    def forward(self, x):
        x = self.dropout1(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(F.relu(self.conv3(x))))

        x = x.view(-1, 64*4*4)

        x = self.dropout4(F.relu(self.fc1(x)))
        x = self.dropout5(F.relu(self.fc2(x)))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

    def predict(self, x):
        img = np.array(x, dtype=np.float32)
        img = torch.from_numpy(img).reshape((1, 1, 28, 28))
        with torch.no_grad():
            output = model.forward(img)

        probablities = F.softmax(output, dim=1)
        top_ps, top_categories = probablities.topk(10, dim=1)

        results = []
        for percent, idx in zip(top_ps, top_categories):
            results.append([categories[idx], percent])

        return results


model = NeuralNet()
model.load_state_dict(torch.load('./model_weights.pt'))
