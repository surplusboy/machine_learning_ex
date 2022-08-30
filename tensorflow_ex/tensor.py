import tensorflow as tf

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

텐서1 = tf.constant([3, 4, 5]) # 파이썬의 숫자, 리스트 등의 데이터들을 담는 곳, 행렬로 값을 저장할 수 있다.
텐서2 = tf.constant([6, 7, 8])
텐서3 = tf.constant([[1, 2], # 텐서로 행렬 표현
                   [3, 4]])
print(tf.add(텐서1, 텐서2))
print(tf.subtract(텐서1, 텐서2))
print(tf.multiply(텐서1, 텐서2))
print(tf.divide(텐서1, 텐서2))

텐서4 = tf.zeros(10) # 길이만 지정한 빈 배열 생성
텐서5 = tf.zeros( [2, 3, 4] ) # 4개의 데이터를 담은 행렬 세개를 두개 생성 (뒤에서부터 읽으면 편함)

print(텐서5.shape)

w = tf.Variable(1) # weight 생성
print(w)
w.assign(2)
print(w)