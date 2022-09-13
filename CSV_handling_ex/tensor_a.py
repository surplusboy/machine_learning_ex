import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

# 사망확률 예측하기

data = pd.read_csv(r'C:\Users\babymon\Desktop\데이터셋\death-rate\train.csv')
# print(data)

print(data.isnull().sum())

# 데이터 전처리
age_avg = round(data['Age'].mean())
embarked_mode_value = data['Embarked'].mode().values[0]
# print(age_avg)
# print(embarked_mode_value)


data['Age'].fillna(value=age_avg, inplace=True)
data['Embarked'].fillna(value=embarked_mode_value,inplace=True)

print(data.isnull().sum())

# X 데이터에 list가 아닌 dict 넣기
y_data = data.pop('Survived')

data_set = trainX, valX, trainY, valY = train_test_split(data, y_data, test_size=0.9, random_state=42)

ds = tf.data.Dataset.from_tensor_slices((dict(trainX), trainY)) # 트레이닝용 데이터셋
ds_val = tf.data.Dataset.from_tensor_slices((dict(valX), valY)) # 검증용 데이터셋

# for i, l in ds.take(1):
#     print(i, l)

'''
각 칼럼의 전처리를 고민
정수로 들어갈 칼럼 : Fare, Parch, Sibsp : tf.feature_column.numeric_column
묶어서 카테고리화 칼럼 : Age : tf.feature_column.bucketized_column
카테고리화 칼럼 : Sex, Embarked, Pclass : tf.feature_column.indicator_column
종류가 너무 다양한 칼럼 : Ticket : tf.feature_column.embedding_column
제외 될 칼럼 : PassengerId, Name
정답 칼럼 : Survived
'''

# def normalizer_fn(x, columns:str):
#     min_value = data[columns].min()
#     max_value = data[columns].max()
#
#     return (x- min_value) / (max_value - min_value)

def normalizer_fn(x):
    min_value = data['Fare'].min()
    max_value = data['Fare'].max()

    return (x - min_value) / (max_value - min_value)

print(tf.feature_column.numeric_column('Age'))
print(normalizer_fn)
# input('break')

# feature columns : 딕셔너리 형태의 데이터를 처리
feature_columns = list()
age = tf.feature_column.numeric_column('Age')
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[10, 20, 30, 40, 50, 60])
# mode = str()

# a = tf.feature_column.numeric_column('Fare')
# print(type(a.key))
print(tf.feature_column.numeric_column('Fare').key)
# input('break')
# input('break')
# exit()

# append_list = [tf.feature_column.numeric_column('Fare',normalizer_fn= lambda x:(x - data[tf.feature_column.numeric_column('Fare').key].min()) / (data[tf.feature_column.numeric_column('Fare').key].max() - data[tf.feature_column.numeric_column('Fare').key].max())),
#                tf.feature_column.numeric_column('Parch'), tf.feature_column.numeric_column('SibSp'), age_bucket, ]

append_list = [tf.feature_column.numeric_column('Fare', normalizer_fn= normalizer_fn),
               tf.feature_column.numeric_column('Parch'), tf.feature_column.numeric_column('SibSp'), age_bucket, ] # for문 안에 feature_columns.append() 넣기

indicator_list = ['Sex', 'Embarked', 'Pclass']
second_append_list = list()


def columns_preprocess(columns, data, mode:str):

    if mode == 'indicator':
        vocab = data[columns].unique()
        cat = tf.feature_column.categorical_column_with_vocabulary_list(columns, vocab)
        one_hot = tf.feature_column.indicator_column(cat)
        locals()['{}_oen_hot'.format(columns.lower())] = one_hot
        return locals()['{}_oen_hot'.format(columns.lower())]
    elif mode == 'embedding':
        vocab = data[columns].unique()
        cat = tf.feature_column.categorical_column_with_vocabulary_list(columns, vocab)
        one_hot = tf.feature_column.embedding_column(cat, dimension=9) # 차원 설정
        locals()['{}_oen_hot'.format(columns.lower())] = one_hot
        return locals()['{}_oen_hot'.format(columns.lower())]


for i in indicator_list:
    second_append_list.append(columns_preprocess(i, data, 'indicator'))

second_append_list.append(columns_preprocess('Ticket', data, 'embedding'))

feature_columns = append_list + second_append_list

# print(type(globals()))
# for i, j in globals().items():
#     try:
#         print(i, j)
#     except Exception as e:
#         print(e)

model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# DenseFeatures 에 input 되는 data 출력해보기
ds_batch = ds.batch(32)
ds_batch_val = ds_val.batch(32)
next(iter(ds_batch))[0]

feature_layer = tf.keras.layers.DenseFeatures( tf.feature_column.numeric_column('Fare'))
feature_layer(next(iter(ds_batch))[0])

model.fit(ds_batch, validation_data=(ds_batch_val), shuffle=True, epochs=50)
