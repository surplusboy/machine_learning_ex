import tensorflow as tf
import os
import numpy as np
import time
import pydot
import graphviz
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

start_time = time.time()

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape( (trainX.shape[0], 28, 28, 1) ) # 데이터 전처리
testX = testX.reshape( (testX.shape[0], 28, 28, 1) )

# 1. 모델 생성l
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1) ), # convolution layer 추가 32개의 커널을 이용해 특성 추출, 3x3 사이즈, Conv2D는 4차원 데이터의 입력 필요 (ndim 에러), color 이미지 일땐 1이 3으로
#     tf.keras.layers.MaxPool2D((2,2 )), # 2x2 사이즈, 최댓값으로 pooling
#     tf.keras.layers.Flatten(), # 1차원 행렬로 압축해주는 레이어, 다차원의 데이터를 단순하게 1차원으로 압축시키게 될 경우, 모델의 예측 능력이 떨어진다. (창의력, 응용력이 낮아짐)
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax') # 마지막 레이어 노드수를 카테고리 갯수만큼 설정하여 확률 예측, sigmoid 는 binary 예측, softmax는 카테고리 예측에 사용
# ])

# model.summary()
# # exit()
#
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # categorical_crossentropy : trainY가 원핫인코딩 되어있을때, sparse_categorical_crossentropy : trainY가 정수로 되어 있을때
#
# # model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10)
#
# model.summary()

# tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# Functional API
input1 = tf.keras.layers.Input(shape=[28, 28])
flatten1 = tf.keras.layers.Flatten()(input1)
dense1 = tf.keras.layers.Dense(28*28, activation='relu')(flatten1)
reshape1 = tf.keras.layers.Reshape((28, 28))(dense1) # reshape : 이전 레이어와 총 node 수가 같아야함

concat1 = tf.keras.layers.Concatenate()([input1, reshape1])
flatten2 = tf.keras.layers.Flatten()(concat1)
output = tf.keras.layers.Dense(10, activation='softmax')(flatten2)

model = tf.keras.Model(input1, output)


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # categorical_crossentropy : trainY가 원핫인코딩 되어있을때, sparse_categorical_crossentropy : trainY가 정수로 되어 있을때

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)