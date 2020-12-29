from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

categories = sorted(['airplane', 'mailbox', 'fish', 'face', 'bowtie', 'butterfly', 'umbrella', 'syringe', 'star', 'elephant', 'hammer', 'key',  'knife', 'ice_cream', 'hand', 'flower', 'fork', 'wheel', 'wine_glass', 'cloud', 'microphone', 'cat', 'baseball', 'crab', 'crocodile',
                     'dolphin', 'ant', 'anvil', 'apple', 'axe', 'banana', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'mushroom', 'octopus', 'screwdriver', 'shark', 'sheep', 'shoe',  'snake',  'snowflake', 'snowman', 'spider', 'camera', 'campfire', 'candle', 'cannon', 'car'])


@app.route("/")
def index():
    return render_template("index.html", classes=categories)
