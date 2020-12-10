from flask import Flask, request, render_template, jsonify
from NeuralNet import model

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.get_json(force=True)['image']
    error = False

    try:
        results = model.predict(image_data)
    except:
        print('An error occured while predicting value.')
        error = True

    return jsonify(results=results, error=error)
