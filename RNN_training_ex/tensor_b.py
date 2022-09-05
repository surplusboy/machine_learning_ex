import numpy as np
import tensorflow as tf
import random

# tensor_a 에서 학습한 모델 적용시키기

Pmodel = tf.keras.models.load_model('./model_repository')


with open(r'./dataset/PIANOBAC.txt', 'r') as text:
    text = text.read()
    bag_of_words = list(set(text))
    bag_of_words.sort()


# utilities 둘 다 만들어놓는게 용이함
text_to_num = {}
num_to_text = {}

for i, data in enumerate(bag_of_words):
    text_to_num[data] = i
    num_to_text[i] = data

new_text = list()

for i in text:
    new_text.append(text_to_num[i])
    
first_input = new_text[117:117+25]
first_input = tf.one_hot(first_input, 31) # 원핫 인코딩
first_input = tf.expand_dims(first_input, axis=0) # 차원 증가

# print(first_input)

# predict_value = Pmodel.predict(first_input)
# predict_value = np.argmax(predict_value[0])
# print(predict_value)
# print(new_text[117+25])

# 연속 예측 하기
music = list()


for i in range(350):

    predict_value = Pmodel.predict(first_input)
    old_predict_value = np.argmax(predict_value[0])
    new_predict_value = np.random.choice(bag_of_words, 1, p=predict_value[0])
    new_predict_value = text_to_num[new_predict_value[0]]
    random_value = random.choice([old_predict_value, new_predict_value])

    # print(new_predict_value)


    # input('break')

    music.append(random_value)
    next_input = first_input.numpy()[0][1:]

    predict_value_one_hot = tf.one_hot(random_value, 31)

    first_input = np.vstack([next_input, predict_value_one_hot.numpy()])
    first_input = tf.expand_dims(first_input, axis=0)

# print(music)

music_text = list()

for i in music:
    music_text.append(num_to_text[i])

print(''.join(music_text))