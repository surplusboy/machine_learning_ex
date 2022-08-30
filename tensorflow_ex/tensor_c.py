import tensorflow as tf
import pandas as pd
import numpy as np
import time

start_time = time.time()

# 3. 데이터 전처리하기 (중간중간 빈 값들을 평균값 기입, 행삭제 등으로 전처리 하여야한다.
data = pd.read_csv('./gpascore.csv')
print(data.isnull().sum()) # 빈값 체크
# data = data.dropna() # NaN/빈값이 존재하는 행 삭제

gre_avg = int(data['gre'].mean())

data = data.fillna(gre_avg) # 빈값이 존재하는 행 채우기
print(data.isnull().sum())

x데이터 = list()
y데이터 = data['admit'].values

for i, rows in data.iterrows():
    # dump데이터 = list()
    x데이터.append([ rows['gre'],
    rows['gpa'],
    rows['rank'] ])
    # for j in rows[1:]:
    #     print(rows)
    #     dump데이터.append(j)
    # x데이터.append(dump데이터)

# del dump데이터
# exit()

# 1. model 생성
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'), # 레이어 내 노드의 갯수 및 활성함수 설정
    tf.keras.layers.Dense(64, activation='tanh'), # 관습적으로 2의 제곱수로 지정한다.
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'), # 마지막 출력 레이어가 될 노드, 0~1 사이의 값을 출력하는 sigmoid를 사용하여 확률 예측
])

# 2. model compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # crossentropy 는 확률 예측에 주로 쓰이는 loss function

# 4. model 학습(fit)
model.fit(np.array(x데이터), np.array(y데이터), epochs=2000) # 첫 인자 : 학습 데이터, 두번째 인자 : 정답 데이터, epoch : 학습 사이클 횟수

# 5. 예측
예측값 = model.predict( [ [750, 3.70, 3], [400, 2.2, 1] ])
# 예측값 = model.predict()
print(예측값)


print("working time : {:.2f} sec:".format(time.time() - start_time))