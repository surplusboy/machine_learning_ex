import os
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt

def 전처리함수(i, 정답):
    i = tf.cast(i/255.0, tf.float32)
    return i, 정답

# 개, 고양이 분류 해보기
data_set_path = r'C:\Users\babymon\Desktop\데이터셋\dogs-vs-cats-redux-kernels-edition'
data_set_flag =['train', 'validation', 'test']
data_set_list = list()

# 이 부분은 함수로 변경하기
a_list = os.listdir(data_set_path)

for i in a_list:
    if os.path.isdir(f'{data_set_path}/{i}') and i in data_set_flag:
        print(f'{i} : {len(os.listdir(f"{data_set_path}/{i}"))}')
        data_set_list.append(f"{data_set_path}/{i}")


# 정렬 알고리즘으로 속도 올리기
# for i in os.listdir(data_set_list[1]):
#     if 'dog' in i:
#         shutil.move(f'{data_set_list[1]}/{i}', f'{data_set_path}/dog')
#     elif 'cat' in i:
#         shutil.move(f'{data_set_list[1]}/{i}', f'{data_set_path}/cat')

# 데이터 전처리 하기 ((x데이터), (y데이터)) 의 shape으로 저장
train_ds = tf.keras.preprocessing.image_dataset_from_directory(f'{data_set_path}/dataset/', image_size=(64, 64), batch_size=64, subset='training', validation_split=0.2, seed=1234) # 데이터셋 경로, 이미지 리사이징, batch 지정(이미지 64개씩), 20% 만큼 vali 데이터 세팅
val_ds = tf.keras.preprocessing.image_dataset_from_directory(f'{data_set_path}/dataset/', image_size=(64, 64), batch_size=64, subset='validation', validation_split=0.2, seed=1234) # 동일한 시드값을 사용함으로써 training, validation 데이터 분류

train_ds = train_ds.map(전처리함수)
val_ds = val_ds.map(전처리함수)

# print(train_ds)
# print(val_ds)
#
for i, 정답 in train_ds.take(1):
    print(i, 정답)
#     plt.imshow(i[0].numpy().astype('uint8'))
#     plt.show()

model = tf.keras.Sequential([
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
    tf.keras.layers.Dense(1, activation='sigmoid') # 마지막 레이어 노드수를 카테고리 갯수만큼 설정하여 확률 예측, sigmoid 는 binary 예측, softmax는 카테고리 예측에 사용
])

# 모델 아웃라인 출력
model.summary()
# exit()

# 2. 모델 compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # categorical_crossentropy : trainY가 원핫인코딩 되어있을때, sparse_categorical_crossentropy : trainY가 정수로 되어 있을때

# 3. 모델 학습
model.fit(train_ds, validation_data=val_ds, epochs=5) # 데이터의 양과 질이 부족하여 최대 accuracy는 85~90% 가량