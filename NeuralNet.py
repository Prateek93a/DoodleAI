import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from helpers import load_categories
from torchvision import transforms

categories = sorted(['airplane', 'mailbox', 'fish', 'face', 'bowtie', 'butterfly', 'umbrella', 'syringe', 'star', 'elephant', 'hammer', 'key',  'knife', 'ice_cream', 'hand', 'flower', 'fork', 'wheel', 'wine_glass', 'cloud', 'microphone', 'cat', 'baseball', 'crab', 'crocodile',
                     'dolphin', 'ant', 'anvil', 'apple', 'arm', 'axe', 'banana', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'mushroom', 'octopus', 'screwdriver', 'shark', 'sheep', 'shoe',  'snake',  'snowflake', 'snowman', 'spider', 'camera', 'campfire', 'candle', 'cannon', 'car'])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5),
                                 (0.5))])

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)

        self.dropout1 = nn.Dropout2d(0.3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.3)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.3)
        self.dropout5 = nn.Dropout(0.4)
        self.dropout6 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(128*2*2, 256)
        # self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, len(categories))

    def forward(self, x):
        x = self.dropout1(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(F.relu(self.conv3(x))))
        x = self.dropout4(self.pool4(F.relu(self.conv4(x))))

        x = x.view(-1, 128*2*2)

        x = self.dropout5(F.relu(self.fc1(x)))
        # x = self.dropout6(F.relu(self.fc2(x)))
        x = self.fc3(x)
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
        raw_x = np.array(x, dtype=np.float32)
        transformed_x = self.transforms(img).view(1, 1, 28, 28)
        return transformed_x


model = Net()
model.load_state_dict(torch.load('./model_weights.pt',
                                 map_location=torch.device('cpu')))
