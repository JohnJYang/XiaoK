import requests
import json

address = eval(input("External IP address: "))

print("老师:同学早上好！我们今天要聊些什么啊？")
question = input("学生:")

url = "http://" + address + ":5000/predict"
headers = {"content-type": "text/text; charset=utf-8"}

response = requests.post(url, data=question.encode('utf-8'), headers=headers)
predictions = response.json()

print(predictions)
