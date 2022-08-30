import tensorflow as tf

train_x = [i for i in range(1,8)]
train_y = [i for i in range(3, 16, 2)]

print(train_x)
print(train_y)

# 관습적으로 weight 는 randomize 하는게 일반적이다
a = tf.Variable(0.1)
b = tf.Variable(0.1)


# 평균제곱오차 (mean squared error)
def 손실함수(a , b):
    예측_y = train_x * a + b # 1. 예측 모델 만들기
    return tf.keras.losses.mse(train_y, 예측_y) # 알고리즘 : (예측1 - 실제2)^2 + (예측2 - 실제2)^2... / 총 갯수


opt = tf.keras.optimizers.Adam(learning_rate=1) # 하이퍼 파라미터


for i in range(2000): # 3. 학습하기
    opt.minimize(lambda:손실함수(a, b), var_list=[a, b]) # 람다 문법을 이용
    # 2. optimizer 및 loos function 지정
    print(a.numpy(), b.numpy())