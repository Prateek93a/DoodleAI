from flask import Flask, request, render_template, json, jsonify
import pickle
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.get_json(force=True)['image']
    img = np.array(image_data, dtype=np.float32)
    img = torch.from_numpy(img).reshape(1, 784)
    with torch.no_grad():
        logits = model.forward(img)

    ps = F.softmax(logits, dim=1)
    top_p, top_class = ps.topk(1, dim=1)
    # print(ps)
    #image_data = np.array(image_data, dtype=float)
    #image_data = image_data.reshape(1, 784)
    #image_data = scaler.fit_transform(image_data)
    #prediction = int(model.predict(image_data)[0])
    # print(prediction)
    return jsonify(number=top_class[0].item())
