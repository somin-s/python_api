from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd 
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route("/getModel", methods=['GET'])
def index():
    return {"ML": "firstModel"}


if __name__ == "__main__":
    app.run(debug=True)
