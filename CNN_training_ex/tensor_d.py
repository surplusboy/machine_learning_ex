import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import shutil
import matplotlib.pyplot as plt

# 모델에 넣기 전에 이미지 증강(변형) 시켜보기

'''
전통적인 방식의 이미지 증강 방법

생성기 = ImageDataGenerator(
    rescale=1./255,
    ratation_range=20, # 회전
    zoom_range=0.15, # 확대
    width_shift_range=0.2, # 이동
    height_shift_range=0.2,
    shear_range=0.15, # 굴절
    horizontal_flip=True, # 가로반전
    fill_mode='nearest')

트레이닝용 = 생성기.flow_from_directory(
    'train 데이터셋 경로',
    class_mode = 'binary', # binary, categorical
    shuffle = Ture,
    seed =123,
    color_mode = 'rgb',
    batch_size = 64,
    target_size = (64, 64),
)

생성기2 = ImageDataGenerator(
    rescale=1./255)
    
검증용 = 생성기2.flow_from_directory(
    'val 데이터셋 경로',
    class_mode = 'binary', # binary, categorical
    shuffle = Ture,
    seed =123,
    color_mode = 'rgb',
    batch_size = 64,
)

model.fit(트레이닝용, validation_data=검증용
'''

def 전처리함수(i, 정답):
    i = tf.cast(i/255.0, tf.float32)
    return i, 정답
#
# # 개, 고양이 분류 해보기
data_set_path = r'C:\Users\babymon\Desktop\데이터셋\dogs-vs-cats-redux-kernels-edition'
data_set_flag =['train', 'validation', 'test']
data_set_list = list()


train_ds = tf.keras.preprocessing.image_dataset_from_directory(f'{data_set_path}/dataset/', image_size=(64, 64), batch_size=64, subset='training', validation_split=0.2, seed=1234) # 데이터셋 경로, 이미지 리사이징, batch 지정(이미지 64개씩), 20% 만큼 vali 데이터 세팅
val_ds = tf.keras.preprocessing.image_dataset_from_directory(f'{data_set_path}/dataset/', image_size=(64, 64), batch_size=64, subset='validation', validation_split=0.2, seed=1234) # 동일한 시드값을 사용함으로써 training, validation 데이터 분류
#
train_ds = train_ds.map(전처리함수)
val_ds = val_ds.map(전처리함수)
#
# for i, 정답 in train_ds.take(1):
#     print(i, 정답)

model = tf.keras.Sequential([
    # 이미지 증강
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(64, 64, 3)), # 뒤집기
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1), # 돌리기
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1), # 축소 확대

    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
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

# 콜백함수 = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint/mnist{epoch}', monitor='val_acc', mode='max', save_weights_only=True, save_freq='epoch') # epoch 끝날때마다 val_acc 값이 max 인 weight값 저장
# 콜백함수 = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint/mnist', monitor='val_acc', mode='max', save_weights_only=True, save_freq='epoch') # epoch 끝날때마다 val_acc 값이 max 인 weight값 저장

# 모델 아웃라인 출력
model.summary()

# 2. 모델 compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # categorical_crossentropy : trainY가 원핫인코딩 되어있을때, sparse_categorical_crossentropy : trainY가 정수로 되어 있을때

# 3. 모델 학습
# model.fit(train_ds, validation_data=val_ds, epochs=12, callbacks=[콜백함수])
model.fit(train_ds, validation_data=val_ds, epochs=10)

