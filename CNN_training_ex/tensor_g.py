import os
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt
import time

# 우수한 성능을 냈던 VGG, ResNet, AlexNet, GoogleNet/Inception 등은 논문등이 공개되어 있다.
# 전이 학습 :성능이 좋은 다른 모델을 가지고 내 모델을 학습시키는 학습

inception_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None) # 1. 구글 Inception 모델 임포트
inception_model.load_weights(r'C:\Users\babymon\Desktop\model\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5') # 2. inception 모델 weights 로드

# inception_model.summary()

# 3. 학습 금지 설정
for i in inception_model.layers:
    i.trainable = False

# Fine Tuning
unfreeze = False
for i in inception_model.layers:
    if i.name == 'mixed6':
        unfreeze = True

    if unfreeze == True:
        i.trainable = True


# 4. 원하는 레이어만 추출
마지막레이어 = inception_model.get_layer('mixed7')
print(inception_model.input)

print(마지막레이어)
print(마지막레이어.output)
print(마지막레이어.output_shape)

# 5. 레이어 연결
layer1 = tf.keras.layers.Flatten()(마지막레이어.output)
layer2 = tf.keras.layers.Dense(1024, activation='relu')(layer1)
drop1 = tf.keras.layers.Dropout(0.2)(layer2)
layer3 = tf.keras.layers.Dense(1, activation='sigmoid')(drop1)

model = tf.keras.Model(inception_model.input ,layer3)

# inception_model.summary()
# exit()

def 전처리함수(i, 정답):
    i = tf.cast(i/255.0, tf.float32)
    return i, 정답
#
# # 개, 고양이 분류 해보기
data_set_path = r'C:\Users\babymon\Desktop\데이터셋\dogs-vs-cats-redux-kernels-edition'
#
# # 데이터 전처리 하기 ((x데이터), (y데이터)) 의 shape으로 저장
train_ds = tf.keras.preprocessing.image_dataset_from_directory(f'{data_set_path}/dataset/', image_size=(150, 150), batch_size=64, subset='training', validation_split=0.2, seed=1234) # 데이터셋 경로, 이미지 리사이징, batch 지정(이미지 64개씩), 20% 만큼 vali 데이터 세팅
val_ds = tf.keras.preprocessing.image_dataset_from_directory(f'{data_set_path}/dataset/', image_size=(150, 150), batch_size=64, subset='validation', validation_split=0.2, seed=1234) # 동일한 시드값을 사용함으로써 training, validation 데이터 분류
#
train_ds = train_ds.map(전처리함수)
val_ds = val_ds.map(전처리함수)


# 모델 아웃라인 출력
model.summary()

# 2. 모델 compile
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=['acc']) # categorical_crossentropy : trainY가 원핫인코딩 되어있을때, sparse_categorical_crossentropy : trainY가 정수로 되어 있을때

tf.keras.utils.plot_model(model, to_file='inception.png', show_shapes=True, show_layer_names=True)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format('Inception' + str(int(time.time())))) # tensorboard --logdir [디렉토리]

# 3. 모델 학습
model.fit(train_ds, validation_data=val_ds, epochs=2, callbacks=[tensorboard]) # 데이터의 양과 질이 부족하여 최대 accuracy는 85~90% 가량