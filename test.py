'''
@Time       : 
@Author     : Jingsen Zheng
@File       : test
@Brief      : 
'''

import torch

x = torch.rand(1, 2, 3, 4)

print(x)
print(x.shape)

# y = x.repeat(2, 2, 2, 2)
# print(y.shape)
# print(y)

sample_point = x[:, :, [0, 2], [0, 3]]
print(sample_point)

x = torch.rand(2, 1)
y = torch.rand(2, 1)

print(x, y)
print(x * y)