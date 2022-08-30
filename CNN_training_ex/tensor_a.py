import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np

start_time = time.time()

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

print(trainX.shape)
print(testX)

trainX = trainX.reshape( (trainX.shape[0], 28, 28, 1) ) # 데이터 전처리
testX = testX.reshape( (testX.shape[0], 28, 28, 1) )

# print(dir(trainY))

plt.imshow( trainX[0] )
plt.gray()
plt.colorbar()
plt.show()

# 1. 모델 생성l
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1) ), # convolution layer 추가 32개의 커널을 이용해 특성 추출, 3x3 사이즈, Conv2D는 4차원 데이터의 입력 필요 (ndim 에러), color 이미지 일땐 1이 3으로
    tf.keras.layers.MaxPool2D( (2,2 )), # 2x2 사이즈, 최댓값으로 pooling
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    # tf.keras.layers.Dense(128, input_shape=(28, 28), activation='relu'), # relu : 음수는 제거하는 활성함수, convolution layer에 주로 사용
    tf.keras.layers.Flatten(), # 1차원 행렬로 압축해주는 레이어, 다차원의 데이터를 단순하게 1차원으로 압축시키게 될 경우, 모델의 예측 능력이 떨어진다. (창의력, 응용력이 낮아짐)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),

    # 해결책 : convolutional layer
    # 1. 이미지에서 중요한 정보를 추려 복사본 20장을 만듬
    # 2. 그곳에는 이미지의 중요한 feature(특성)이 담겨있음
    # 3. feature extraction 학습 -> 이미지의 중요한 특성을 kernel 을 거쳐 압축 시킴
    # 4. translation invariance
    tf.keras.layers.Dense(10, activation='softmax') # 마지막 레이어 노드수를 카테고리 갯수만큼 설정하여 확률 예측, sigmoid 는 binary 예측, softmax는 카테고리 예측에 사용
])

# 모델 아웃라인 출력
model.summary()
# exit()

# 2. 모델 compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # categorical_crossentropy : trainY가 원핫인코딩 되어있을때, sparse_categorical_crossentropy : trainY가 정수로 되어 있을때

# 3. 모델 학습
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10)

# 4. 학습 후 모델 평가
score = model.evaluate( testX, testY )
print(score) # loss와 accuracy를 출력


# 5. 예측
# 예측값 = model.predict( textX )
# print(예측값)
print("working time : {:.2f} sec:".format(time.time() - start_time))

#