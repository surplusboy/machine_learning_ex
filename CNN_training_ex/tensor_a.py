import tensorflow as tf
import matplotlib.pyplot as plt
import time

start_time = time.time()

(trainX, trainY), (textX, textY) = tf.keras.datasets.fashion_mnist.load_data()

print(trainX.shape)
print(textX)

# print(dir(trainY))

plt.imshow( trainX[0] )
plt.gray()
plt.colorbar()
plt.show()
input('break')

# 1. 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(28, 28), activation='relu'), # 음수는 제거하는 활성함수, convolution layer에 주로 사용
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Flatten(), # 1차원 행렬로 압축해주는 레이어, 다차원의 데이터를 단순하게 1차원으로 압축시키게 될 경우, 모델의 예측 능력이 떨어진다. (창의력, 응용력이 낮아짐)
    # 해결책 : convolutional layer
    # 1. 이미지에서 중요한 정보를 추려 복사본 20장을 만듬
    # 2. 그곳에는 이미지의 중요한 feature(특성)이 담겨있음
    # 3. feature extraction 학습
    tf.keras.layers.Dense(10, activation='softmax') # 마지막 레이어 노드수를 카테고리 갯수만큼 설정하여 확률 예측, sigmoid 는 binary 예측, softmax는 카테고리 예측에 사용
])

# 모델 아웃라인 출력
model.summary()
# exit()

# 2. 모델 compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # categorical_crossentropy : trainY가 원핫인코딩 되어있을때, sparse_categorical_crossentropy : trainY가 정수로 되어 있을때

# 3. 모델 학습
model.fit(trainX, trainY, epochs=5)


# 4. 예측
# 예측값 = model.predict( textX )
# print(예측값)
print("working time : {:.2f} sec:".format(time.time() - start_time))

#