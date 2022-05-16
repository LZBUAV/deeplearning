import sys
import os
file_path = os.path.dirname(__file__)
parent_path = os.path.dirname(file_path)
sys.path.append(parent_path)
from lib.bbnet import Perceptioin
from lib.activate_function import ActivateFun
from lib.data_sets import and_data

datas = and_data()
input_vectors = datas.samples
labels = datas.labels
print("inputs : ", input_vectors)
print("labels : ", labels)

active_fun = ActivateFun()
f = active_fun.step_fun

percerpion = Perceptioin(2, f)
percerpion.train(input_vectors, labels, 0.1, 10)

print("1 and 0 = ", percerpion.forward([1,0]))
print("1 and 1 = ", percerpion.forward([1,1]))
print("0 and 1 = ", percerpion.forward([0,1]))
print("0 and 0 = ", percerpion.forward([0,0]))

print(percerpion)