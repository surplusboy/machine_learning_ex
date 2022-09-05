import numpy as np
import tensorflow as tf
import time
start_time = time.time()

with open(r'./dataset/PIANOBAC.txt', 'r') as text:
    text = text.read()
    bag_of_words = list(set(text))
    bag_of_words.sort()

    print(bag_of_words)

# utilities 둘 다 만들어놓는게 용이함
text_to_num = {}
num_to_text = {}

for i, data in enumerate(bag_of_words):
    print(data)
    text_to_num[data] = i
    num_to_text[i] = data
    

new_text = list()

for i in text:
    new_text.append(text_to_num[i])
print(new_text)

x = list()
y = list()

print(len(new_text))
for i in range(len(new_text)-25):
    x.append(new_text[i:i+25])
    y.append(new_text[i+25])


print(x[0])
print(y[0])

print(np.array(x).shape)

# 원핫인코딩 (데이터가 많을땐 embedding layer)
x = tf.one_hot(x, 31)
y = tf.one_hot(y, 31)

print(x.shape[1])

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(x.shape[1], x.shape[2])), # LSTM 중첩 시엔 return_sequences=True 파라미터 필요
    tf.keras.layers.Dense(x.shape[2], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(x, y, batch_size=64, epochs=100, verbose=2)
model.fit(x, y, batch_size=64, epochs=100)

model.save(r'./model_repository')

print("working time : {:.2f} sec:".format(time.time() - start_time))