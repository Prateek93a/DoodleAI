from flask import Flask, request, render_template, json, jsonify
import pickle
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))
classes = ['airplane', 'alarm', 'brain', 'butterfly', 'car',
           'cat', 'crab', 'face', 'giraffe']


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        # self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        # self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        # self.conv6 = nn.Conv2d(128, 128, 3, padding=1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32*7*7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.dropout1(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.relu(self.conv2(x))))
        # x = (self.pool(F.relu(self.conv2(x))))

        # x = (F.relu(self.conv3(x)))
        # x = (self.pool(F.relu(self.conv4(x))))

        # x = (F.relu(self.conv5(x)))
        # x = self.pool(F.relu(self.conv6(x)))

        x = x.view(-1, 32*7*7)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


model = Net()
model.load_state_dict(torch.load('model_cifar.pt'))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.get_json(force=True)['image']
    img = np.array(image_data, dtype=np.float32)
    img = torch.from_numpy(img).reshape((1, 1, 28, 28))
    with torch.no_grad():
        logits = model.forward(img)

    ps = F.softmax(logits, dim=1)
    top_p, top_class = ps.topk(1, dim=1)
    return jsonify(label=classes[top_class[0].item()])
