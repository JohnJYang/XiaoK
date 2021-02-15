import tensorflow_hub as hub
import tensorflow as tf

from gpt2_tokenizer import GPT2Tokenizer

print(tf.__version__)
print(hub.__version__)
print("GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tokenizer = GPT2Tokenizer(
    'CPM-Generate/bpe_3w_new/vocab.json',
    'CPM-Generate/bpe_3w_new/merges.txt',
    model_file='CPM-Generate/bpe_3w_new/chinese_vocab.model')

gpt = hub.load('./cpm-lm-tf2_v2/')

text = f'''以下是一位老师和学生的对话。这位老师会根据学生不同的回答提出不同的问题，以此鼓励学生多交流，把心里想说的话都说出来。
老师:同学早上好！你今天起床之后都做什么了呀？
学生:我去帮我奶奶放了牛，然后再陪我姐姐一起去了超市。
老师:哇好棒哦，你们买了什么呀？
学生:青菜，豆腐，还有一条腊肠。
老师:嗯嗯，最近是快过春节了呢，过年你爸妈会回来吗？
学生:是哒！我妈妈和爸爸都会回家。
老师:太棒啦！你能和我说说关于你父母的事吗？
学生:我妈妈在上海上班，我爸爸在杭州。
老师:嗯嗯，真的很好！所以你一定要好好读书哦！诶对了，你听过关于年的传说吗？
学生:没有诶。你可以和我说说吗？
老师:当然可以！相传很久以前，世界突然出现了一种叫做年兽的怪物。每当新年来临，它就要跑到一个村子来吃人。所以新年临近时，人们都要搬到山里去躲灾。又一年，年兽又回来了。但是，突然，不知从哪里传出“噼里啪啦”的声音，只见一个小孩正在拼命地燃放烟花、爆竹。年兽不知道这是何物，吓得赶紧逃跑。接下来啊，年兽怕烟花爆竹这一消息马上就传开了，于是，每当新年来临，人们都要燃放烟花爆竹，以求得新年平安。
学生:哦！原来如此，怪不得我们过年有很多习俗呢。
老师:是的呀！我们上次聊到你的课间活动，你在学校课间最喜欢做的事是什么呀？
学生:我一般都会和我朋友去操场跳绳。
老师:哇真有趣！我记得我小时候也经常跳绳，你能跟我说说你为什么喜欢跳绳吗？
学生:因为跳绳可以锻炼身体啊！出汗之后就很舒服的。
老师:那你在学校最喜欢上的课是什么呀？
学生:语文课吧。
老师:语文课啊! 你可以和我说说你为什么喜欢呢？
学生:因为我觉得语文课的课本很有意思。我也很喜欢背古诗。
老师:真厉害！你可以给我背一首古诗吗？
学生:当然可以。绝句，唐，杜甫。两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。
老师:好棒好棒！那我们今天就聊到这里吧，明天见！
学生:嗯嗯，明天见！
老师:同学早上好！我们今天要聊些什么呀？
学生:'''

new_question = "老师:同学早上好！我们今天要聊些什么啊？"

def create(tokenizer, gpt, prompt, number, length, top_p, temperature):
    inputs = tf.constant([tokenizer.encode(prompt)] * number, dtype=tf.int64)
    length = tf.constant(length, dtype=tf.int64)
    ret = gpt.signatures['serving_default'](
        inp=inputs,
        length=length,
        top_p=tf.constant(top_p, tf.float32),
        temperature=tf.constant(temperature, tf.float32)
    )['output_0']
    return [tokenizer.decode(s).replace(' ', '')for s in ret.numpy()]


def ask_XiaoK(question, number, length, top_p, temperature):
    global new_question, text
    if new_question:
        print(new_question)
        new_question = ''
    text += question
    ret = create(tokenizer, gpt, text, number, length, top_p, temperature)
    for x in ret:
        a = x[len(text):]
        show = a.split('\n')
        if not show[0]:
            show = show[1]
        else:
            show = show[0]
    text += "\n"
    text += show
    text += "\n学生:"
    return show


while True:
    if new_question:
        print(new_question)
        new_question = ''
    question = input("学生:")
    if question == 'quit' or question == 'exit':
        break
    elif question == 'text':
        print(text)
    else:
        print(ask_XiaoK(question, 1, 50, 0.7, 1))
