from flask import Flask, jsonify, request
from flask_cors import CORS

from nlpmodule.tools.Classifier import classifier_treatment_load

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Bienvenido a API KT-BERT"


@app.route("/predict", methods=['POST'])
def predict():
    json = request.get_json(force=True)
    print('----------------------> ',json)
    print(type(json))
    print(json['text'])

    text = json['text']
    result = classifier_treatment_load(text)
    return result