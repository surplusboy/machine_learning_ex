# from tensorflow.kears.callbacks import TensorBoard
import tensorflow as tf

import matplotlib.pyplot as plt
import time


# 텐서보드 사용해보기

start_time = time.time()

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape( (trainX.shape[0], 28, 28, 1) ) # 데이터 전처리
testX = testX.reshape( (testX.shape[0], 28, 28, 1) )

# print(dir(trainY))

# plt.imshow( trainX[0] )
# plt.gray()
# plt.colorbar()
# plt.show()

# 1. 모델 생성

# default_model_setting = [
#     tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1) ), # convolution layer 추가 32개의 커널을 이용해 특성 추출, 3x3 사이즈, Conv2D는 4차원 데이터의 입력 필요 (ndim 에러), color 이미지 일땐 1이 3으로
#     tf.keras.layers.MaxPool2D((2, 2)), # 2x2 사이즈, 최댓값으로 pooling
#     tf.keras.layers.Flatten(), # 1
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax') # 마지막 레이어 노드수를 카테고리 갯수만큼 설정하여 확률 예측, sigmoid 는 binary 예측, softmax는 카테고리 예측에 사용
# ]

default_model_setting = [
    tf.keras.layers.Dense(10, activation='softmax') # 마지막 레이어 노드수를 카테고리 갯수만큼 설정하여 확률 예측, sigmoid 는 binary 예측, softmax는 카테고리 예측에 사용
]

default_model_setting.insert(0, tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
print(dir(default_model_setting[0]))
# print(default_model_setting[0].get_input_mask_at)
print(default_model_setting[0].get_input_shape_at)

print(default_model_setting)
input('break')

model = tf.keras.Sequential(
    default_model_setting
)

# 모델 아웃라인 출력
model.summary()
input('break')
exit()



# 2. 모델 compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # categorical_crossentropy : trainY가 원핫인코딩 되어있을때, sparse_categorical_crossentropy : trainY가 정수로 되어 있을때


tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format('Conv1time' + str(int(time.time()))))

# 3. 모델 학습
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5, callbacks=[tensorboard])

# 4. 학습 후 모델 평가
score = model.evaluate( testX, testY )
print(score) # loss와 accuracy를 출력


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)), # convolution layer 추가 32개의 커널을 이용해 특성 추출, 3x3 사이즈, Conv2D는 4차원 데이터의 입력 필요 (ndim 에러), color 이미지 일땐 1이 3으로
    tf.keras.layers.MaxPool2D((2, 2)), # 2x2 사이즈, 최댓값으로 pooling
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D((2, 2)), # 2x2 사이즈, 최댓값으로 pooling
    tf.keras.layers.Flatten(), # 1
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') # 마지막 레이어 노드수를 카테고리 갯수만큼 설정하여 확률 예측, sigmoid 는 binary 예측, softmax는 카테고리 예측에 사용
])

# 모델 아웃라인 출력
model.summary()

# 2. 모델 compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # categorical_crossentropy : trainY가 원핫인코딩 되어있을때, sparse_categorical_crossentropy : trainY가 정수로 되어 있을때


tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format('Conv2time' + str(int(time.time()))))

# Early Stopping

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')


# 3. 모델 학습
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5, callbacks=[tensorboard])


print("working time : {:.2f} sec:".format(time.time() - start_time))
