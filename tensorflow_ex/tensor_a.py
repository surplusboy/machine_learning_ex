import tensorflow as tf

# 키와 신발사이즈의 상관관계

키 = 170
신발 = 260

# 신발 사이즈 예측 모델
# 신발 = 키 * a + b

def 손실함수():
    예측값 = 키 * a +b
    손실값 = tf.square(신발 - 예측값) # square : 제곱 메소드
    return 손실값

a = tf.Variable(0.1)
b = tf.Variable(0.2)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(300):
    opt.minimize(손실함수, var_list=[a, b])
    print(a.numpy(), b.numpy())

print(키*1.52+1.62)