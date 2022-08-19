# import tensorflow as tf

# print(tf.__version__)

# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

a = 50
b = 60

h1 = a * 0.2 + b * 0.1
h2 = a * 0.3 + b * 0.2

result = h1 * 0.5 + h2 * 0.4

print(result)