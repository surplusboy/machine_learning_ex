from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

data_set_path = r'C:/Users/babymon/Desktop/데이터셋/사람얼굴/archive/img_align_celeba/img_align_celeba'

images = list()

for i in os.listdir(data_set_path)[0:50000]:
    old_image = Image.open(f'{data_set_path}/{i}').crop((20, 30, 160, 180)).convert('L').resize((64, 64))
    images.append(np.array(old_image))

# plt.imshow(images[0])
# plt.show()

# print(images.shape)

# 이미지 전처리
images = np.divide(images, 255)
images = images.reshape(50000, 64, 64, 1) # 흑백 이미지 4차원으로 증강
# images.reshape( 5 ,)

print(images.shape)

# discriminator 모델 생성
discriminator = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=[64,64,1]),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

noise_shape = 100

# generator 모델 생성
generator = tf.keras.models.Sequential([
  tf.keras.layers.Dense(4 * 4 * 256, input_shape=(noise_shape,)),
  tf.keras.layers.Reshape((4, 4, 256)),
  tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same'), # upsampling2D도 찾아볼것
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
])

generator.summary()

GAN = tf.keras.models.Sequential([generator, discriminator])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.trainable = False

GAN.compile(optimizer='adam', loss='binary_crossentropy')


def predict_pic(time: int, cycle: int):

    show_img = plt
    show_img.figure(f'{str(cycle+1)} 회차 결과')
    predict_value = generator.predict((lambda x, y : np.random.uniform(x, y, size=(20, 100)))(-1, 1))
    # print(predict_value.shape)
    for i in range(20):
        show_img.subplot(4, 5, i+1)
        show_img.imshow(predict_value[i].reshape(64, 64), cmap='gray') # 컬러면 64, 64, 3
        show_img.axis('off')

    show_img.tight_layout()
    show_img.show(block=False)
    show_img.pause(time)
    show_img.close()


x_data = images


for i in tqdm(range(300)):
    print(f'현재 epoch {i+1}회차')
    predict_pic(5, i)

    for j in range(50000//128):
        if j % 100 == 0:
            print(f'현재 batch {j+1}회차')

        # discriminator 트레이닝
        real_images = x_data[j*128:(j+1)*128]
        real_markings = np.ones(shape=(128, 1))
        loss1 = discriminator.train_on_batch(real_images, real_markings) # 진짜 사진

        random_value = np.random.uniform(-1, 1, size=(128, 100))
        fake_images = generator.predict(random_value, verbose=0)
        fake_markings = np.zeros(shape=(128, 1))

        loss2 = discriminator.train_on_batch(fake_images, fake_markings) # 가짜 사진
        
        # real_images 와 fake_images 셔플해서 학습해보기

        # generator 트레이닝
        loss3 = GAN.train_on_batch(random_value, real_markings)

    print(f'이번 epoch 의 최종 loss discriminator loss : {loss1+2}, GAN loss : {loss3}')


'''
더 해봐야할 것 들
GAN 네트워크의 layer들을 수정하고 더해보기 
이미지를 더 사용하거나 살짝 비틀어서 집어넣어보기
label smoothing 같은 잡기술 넣어보기 
noise (랜덤숫자) 다르게 설정해보기 
요즘 GAN은 어떻게 만드나 살펴보기
'''