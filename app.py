from flask import Flask, jsonify, request
from xiaok import ask_XiaoK

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
	question = request.data.decode('utf-8')
	print(question)
	print(text)
	predictions = ask_XiaoK(question, 1, 50, 0.8, 0.75)
	return jsonify(predictions)


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000, debug=True)
