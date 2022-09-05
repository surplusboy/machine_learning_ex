import urllib.request
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

print('데이터 수집 시작')
# urllib.request.urlretrieve('https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt', 'shopping.txt')
# with open(r'C:\Users\babymon\Desktop\데이터셋\naver_shopping.txt', 'r', encoding='utf-8') as text:
#     shopping = pd.read_csv(text, sep = r'\t', engine='python', encoding='cp949')

print('데이터 수집 완료')


# print(shopping)

input('break')

raw = pd.read_table('./shopping.txt', names=['rating', 'review'])

raw['label'] = np.where(raw['rating'] > 3, 1, 0) # 삼항연산자
# print(raw)

# 데이터 전처리
raw['review'] = raw['review'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]', '', regex=True)  # 정규 표현식을 이용해 특수문자 제거, 오타제거는 서치해보기

# print(raw)

print(raw.isnull().sum())
raw.drop_duplicates(subset=['review'], inplace=True)  # 중복 제거

# bag_of_words 생성
bag_of_words = raw['review'].tolist()
bag_of_words = ''.join(bag_of_words)

bag_of_words = list(set(bag_of_words))
bag_of_words.sort()


# 데이터 전처리
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, oov_token='<OOV>') # 문자들을 정수로 변환, False 는 단어 단위를 변환
char_list = raw['review'].tolist()
tokenizer.fit_on_texts(char_list) # 추후엔 val/test 는 빼고 토크나이징

# print(tokenizer.word_index)

train_seq = tokenizer.texts_to_sequences(char_list) # train 데이터셋 전체 숫자 변환

y = raw['label'].tolist()


raw['length'] = raw['review'].str.len()
print(raw.describe()) # 최대 글자 제한 전 통계 확인
print(raw['length'][raw['length'] < 120].count())

x = tf.keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=100) # 모든 문장 길이 제한, 미달시 0으로 fill

trainX, valX, trainY, valY = train_test_split(x, y, test_size=0.2, random_state=42) # 시드는 42가 괍습적

print(len(trainX))
print(len(valX))
# trainX  = tf.expand_dims(trainX, axis=0)
# valX  = tf.expand_dims(valX, axis=0)

print(trainX.shape)

# trainDS = (trainX, trainY)
# valDS = (valX, valY)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 16),
    tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(trainX.shape[0], trainX.shape[1])),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=False),
    # tf.keras.layers.LSTM(100, input_shape=(159540, 39885)),
    # tf.keras.layers.LSTM(512, return_sequences=True),
    # tf.keras.layers.LSTM(256, return_sequences=True),
    # tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="swish"),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(np.array(trainX), np.array(trainY), batch_size=64, epochs=5)
model.fit(np.array(trainX), np.array(trainY), epochs=5)

score = model.evaluate(np.array(valX), np.array(valY))
print(score)
