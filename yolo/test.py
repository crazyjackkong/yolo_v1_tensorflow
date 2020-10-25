import numpy as np
cell_size = 7
boxes_per_cell = 2


'''
'''
# 3.reshape之后再转置，变成7*7*2的三维数组
offset = np.transpose(
    # 2.创建完成后reshape为2*7*7的三维数组
    np.reshape(
        # 1.创建 14*7的二维数组
        np.array(
            [np.arange(cell_size)] * cell_size * boxes_per_cell),
        (boxes_per_cell, cell_size, cell_size)),
    (1, 2, 0))

print(offset)

a = np.transpose(offset,(1,0,2))
print(a)
