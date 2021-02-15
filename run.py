from xiaok import ask_XiaoK

print("老师:同学早上好！我们今天要聊些什么啊？")

while True:
    question = input("学生:")
    if question == 'quit' or question == 'exit':
        break
    elif question == 'text':
        print(text)
    else:
        print(ask_XiaoK(question, 1, 50, 0.7, 1))
