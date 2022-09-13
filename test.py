import matplotlib.pyplot as plt

# a = [1, 2, 3, 4]
# b = [5, 6, 2.3]
#
# a = a + b
#
# print(a)

a = 'string'
# len(a) >>> 6
b = 0

# if not type(a) == str and len(a) > 2:
#     print('test')

# if not len(a) > 2 and b > 1:
#     print('참')

test_dict = {
    'firtst' : 1
}

def test_func(x):
    return x + '\ttest_func'


print((lambda x : test_func(x))('응애'))

plt.f