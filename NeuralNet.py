import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from helpers import load_categories
from torchvision import transforms

categories = sorted(['airplane', 'mailbox', 'fish', 'face', 'bowtie', 'butterfly', 'umbrella', 'syringe', 'star', 'elephant', 'hammer', 'key',  'knife', 'ice_cream', 'hand', 'flower', 'fork', 'wheel', 'wine_glass', 'cloud', 'microphone', 'cat', 'baseball', 'crab', 'crocodile',
                     'dolphin', 'ant', 'anvil', 'apple', 'axe', 'banana', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'mushroom', 'octopus', 'screwdriver', 'shark', 'sheep', 'shoe',  'snake',  'snowflake', 'snowman', 'spider', 'camera', 'campfire', 'candle', 'cannon', 'car'])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)

        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.3)
        self.dropout3 = nn.Dropout2d(0.3)
        self.dropout4 = nn.Dropout2d(0.3)
        self.dropout5 = nn.Dropout(0.5)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 1)

        self.fc1 = nn.Linear(256, len(categories))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.pool1(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout2(self.pool2(x))

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dropout3(self.pool3(x))

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.dropout4(self.pool4(x))

        x = self.avgpool(x)

        x = x.view(-1, 256)

        x = self.dropout5(x)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)

        return x

    def predict(self, x):
        img = self.transform(x)
        with torch.no_grad():
            output = model.forward(img)
            probablities = F.softmax(output, dim=1)
        top_ps, top_categories = probablities.topk(5, dim=1)
        top_ps = top_ps[0].tolist()
        top_categories = top_categories.int()[0].tolist()
        results = []
        for percent, idx in zip(top_ps, top_categories):
            results.append([categories[int(idx)], percent])

        return results

    def transform(self, x):
        raw_x = np.array(x, dtype=np.float32).reshape(28, 28)
        transformed_x = self.transforms(raw_x).view(1, 1, 28, 28)
        return transformed_x


model = Net()
model.load_state_dict(torch.load('./model_weights.pt',
                                 map_location=torch.device('cpu')))
