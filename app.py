from flask import Flask, request, render_template, jsonify
from NeuralNet import model, categories

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", classes=categories)


@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.get_json(force=True)['image']
    error = False
    results = []
    try:
        results = model.predict(image_data)
    except Exception as e:
        print('An error occured while predicting value.', e)
        error = True

    return jsonify(label=results, error=error)
