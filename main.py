import tensorflow as tf
import numpy as np

# print(tf.__version__)

# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# a = 50
# b = 60
#
# h1 = a * 0.2 + b * 0.1
# h2 = a * 0.3 + b * 0.2
#
# result = h1 * 0.5 + h2 * 0.4
#
# print(result)

# test_list = [[1, 2, 3], [4, 5, 6]]
#
# a = np.array([test_list])
# print(a[...])
# print(a)


# test_li =[[1,2,3], [4,5,6]]
# print(test_li[...])
#
# d = [1, 2, 3, 4, 5, 6]
# print(d[1::2])
#
a = 5
y = 3

print((a-y)**2)

print((a**2)- (2*(a*y)) + y**2)


# TensorFlow 는 파이썬 기본 자료형과 조금 다르다는 것을 알 수 있음
case_a_list = [1,2,3,4]
case_a_variable = 2

case_b_list = [1,2,3,4]
case_b_variable = tf.Variable(2)

print(case_a_list*case_a_variable)
print(case_b_list*case_b_variable)
