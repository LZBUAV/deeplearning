import sys
import os
file_path = os.path.dirname(__file__)
parent_path = os.path.dirname(file_path)
sys.path.append(parent_path)

from lib.activate_function import ActivateFun
from lib.bbnet import LinerUnit
from lib.data_sets import liner_unit_data

activate_fun = ActivateFun()
f = activate_fun.liner_fun

datas = liner_unit_data()
input_vectors = datas.input_vecs
labels = datas.labels

liner_unit = LinerUnit(1, f)

liner_unit.train(input_vectors, labels, 0.01, 10)

print(liner_unit)

print("6.3 : ", liner_unit.forward([6.3]))
print("7 : ", liner_unit.forward([7]))
print("13 : ", liner_unit.forward([13]))