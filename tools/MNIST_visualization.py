import os
import sys
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import ptp
file_path = os.path.dirname(__file__)
parent_path = os.path.dirname(file_path)
sys.path.append(parent_path)

#可视化MNIST数据集
f = open(parent_path + '/datasets/MNIST/train-images.idx3-ubyte', 'rb')
content = f.read()
f.close()
for pic in range(60000):
    picture = []
    for i in range(28):
        picture.append([])
        for j in range(28):
            picture[i].append(content[16 + 28*28*pic + 28*i + j])
    plt.ion()
    plt.imshow(picture)
    plt.show()
    plt.show()
    plt.pause(0.5)
    plt.clf()
    print("picture :", pic)
