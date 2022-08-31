import tensorflow as tf

# weight 값만 저장한 모델 불러와보기
data_set_path = r'C:\Users\babymon\Desktop\데이터셋\dogs-vs-cats-redux-kernels-edition'
data_set_flag =['train', 'validation', 'test']
data_set_list = list()

test_ds = tf.keras.preprocessing.image_dataset_from_directory(f'{data_set_path}/dataset/', image_size=(64, 64), batch_size=64)

def 전처리함수(i, 정답):
    i = tf.cast(i/255.0, tf.float32)
    return i, 정답

test_ds = test_ds.map(전처리함수)



model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Dropout(0.2), # over fitting 을 완화할 수 있는 레이어, 과적합을 방지하기 위의 레이어 노드 일부를 제거함
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # 마지막 레이어 노드수를 카테고리 갯수만큼 설정하여 확률 예측, sigmoid 는 binary 예측, softmax는 카테고리 예측에 사용
])

# 모델 아웃라인 출력
model2.summary()

# 모델 compile
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.load_weights('checkpoint/mnist')

model2.evaluate(test_ds)