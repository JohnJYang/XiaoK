from flask import Flask, jsonify, request
from xiaok import ask_XiaoK

app = Flask(__name__)


@app.route(“/predict”, methods=[‘POST’])
def predict():
   predictions = ask_XiaoK(request, 1, 50, 0.8, 0.75)
   return jsonify(predictions)


if __name__ == “__main__”:
   app.run(host='0.0.0.0', port=5000, debug=True)
