import sys
import os
file_path = os.path.dirname(__file__)
parent_path = os.path.dirname(file_path)
sys.path.append(parent_path)

from functools import reduce
from lib import bbnet
from lib import activate_function
from lib import data_sets
import datetime
import numpy as np

#--------------------------------------------------梯度检查--------------------------------------------------------
#计算两个向量之间的误差
def cacl_error(lable, output):
    error = 0.5 * reduce(
        lambda a, b : a + b, list(
            map(
                lambda a : (a[0] - a[1]) * (a[0] - a[1]), list(zip(lable, output))
            )
        ), 0.0
    )
    return error

#计算两个向量之间的误差
def cacl_grad_vector(lable, output):
    return 0.5 * ((lable - output) * (lable - output)).sum()

#根据极限法求梯度来与神经网络计算出的梯度做对比
def grad_check(network, lable, sample):
    network.get_grad(lable, sample)
    e = 0.0001
    for layer in network.layers:
        for node in layer.nodes:
            for conn in node.downstream_nodes:
                actrul_grad = conn.get_grad()
                conn.weight = conn.weight + e
                output = network.predict(sample)
                error1 = cacl_error(lable, output)
                conn.weight = conn.weight - e - e
                output = network.predict(sample)
                error2 = cacl_error(lable, output)
                expect_grad = (error2 - error1) / (e + e)
                print("actrul_grad :", actrul_grad, "\nexpect_grad :", expect_grad, "\nerror       :", actrul_grad - expect_grad)
                print("----------------------------------------------")
                conn.weight = conn.weight + e

#该程序用来检测梯度计算是否正确，从而可以判断神经网络代码实现是否正确，已验证梯度正确
def test_grad():
    sample = [0.9, 0.1, 0.1, 0.9, 0.8, 0.7, 0.6]
    lable = [0.9, 0.1, 0.1, 0.9, 0.8, 0.7, 0.6]
    func = activate_function.ActivateFun()
    f = func.sigmod
    net = bbnet.Fully_Connected_Network([7,30,7],f)
    grad_check(net, lable, sample)

#---------------------------------------------使用自己生成的数据集训练-----------------------------------------------
#该实验效果很差主要原因是数据集太少
#神经网络训练测试1，使用生成的32个输入向量，32个和输入同样的标签，实现y = x
def get_data_set():
    normalizer = data_sets.Normalizer()
    data_set = []
    label = []
    for i in range(0, 256, 8):
        data = normalizer.normal(i)
        data_set.append(data)
        label.append(data)
    return data_set, label

#对训练好的网络计算正确率,面向对象的方式实现网络
def calc_correct_rate(net):
    normalizer = data_sets.Normalizer()
    correct = 0
    for i in range(0,256,8):
        data = normalizer.normal(i)
        output = net.predict(data)
        output = normalizer.denormal(output)
        if output == i:
            correct = correct + 1
    correct_rate = correct / 256
    return correct_rate

#对训练好的网络计算正确率,向量化编程的方式实现网络
def calc_correct_rate_VECTOR(net):
    normalizer = data_sets.Normalizer()
    correct = 0
    for i in range(0,256,8):
        data = normalizer.normal(i)
        data = np.array(data).reshape(8,1)
        output = net.predict(data)
        output = list(output)
        output = normalizer.denormal(output)
        if output == i:
            correct = correct + 1
    correct_rate = correct / 256
    return correct_rate

#测试1训练 可能是由于数据集数量太小，该测试epoch设置低时能达到45%正确率，设置高时过拟合
def train_Y_X():
    data_set, lable = get_data_set()
    activate_func = activate_function.ActivateFun()
    func = activate_func.sigmod
    net = bbnet.Fully_Connected_Network([8,5,8], func)
    net.train(lable, data_set, 0.006, 150000)
    correct_rate = calc_correct_rate(net)
    print("correct_rate :", correct_rate)

#测试1训练 可能是由于数据集数量太小，该测试epoch设置低时能达到45%正确率，设置高时过拟合,向量化编程
def train_Y_X_VECTOR():
    data_set, lable = transpose(get_data_set())
    activate_func = activate_function.ActivateFun()
    func = activate_func.sigmod
    net = bbnet.Fully_Connected_Network_Vector([8,5,8], activate_func)
    net.train(lable, data_set, 0.006, 150000)
    correct_rate = calc_correct_rate(net)
    print("correct_rate :", correct_rate)

#---------------------------------------------使用MINIST数据集训练-----------------------------------------------
#神经网络训练=测试2，使用MINIST数据集
def get_training_data_set():
    image_loader = data_sets.ImageLoader(parent_path + '/datasets/MNIST/train-images.idx3-ubyte', 600)
    label_loader = data_sets.LabelLoader(parent_path + '/datasets/MNIST/train-labels.idx1-ubyte', 600)
    return image_loader.load(), label_loader.load()

def get_test_data_set():
    image_loader = data_sets.ImageLoader(parent_path + '/datasets/MNIST/t10k-images.idx3-ubyte', 100)
    label_loader = data_sets.LabelLoader(parent_path + '/datasets/MNIST/t10k-labels.idx1-ubyte', 100)
    return image_loader.load(), label_loader.load()

#获取结果，对网络输出进行后处理
def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

#使用测试数据集评估训练好的网络的错误了
def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)

#测试2训练函数，使用面向对象的方式实现，速度极慢，一个epoch要2个小时，无法实现训练
def train_MNIST_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    print("reading datas.........")
    train_data_set, train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()
    print("read datas finished!------")
    activate_func = activate_function.ActivateFun()
    func = activate_func.sigmod
    network = bbnet.Fully_Connected_Network([784, 300, 10], func)
    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.005, 1)
        print('%s epoch %d finished' % (datetime.datetime.now(), epoch))
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print('%s after epoch %d, error ratio is %f' % (datetime.datetime.now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio

#---------------------------------------------使用向量式编程重构网络实现-----------------------------------------------
#梯度检查，已经验证梯度正确
def grad_check_vector(network, lable, sample):
    network.predict(sample)
    network.calc_delta(lable)
    e = 0.0001
    for layer in network.layers:
        for i in range(layer.weights.shape[0]):
            for j in range(layer.weights.shape[1]):
                layer.weights[i][j] += e
                output = network.predict(sample)
                error1 = cacl_grad_vector(lable, output)
                layer.weights[i][j] -= 2*e
                output = network.predict(sample)
                error2 = cacl_grad_vector(lable, output)
                expect_grad = (error2 - error1) / (e + e)
                layer.weights[i][j] = layer.weights[i][j] + e
                print("actrul_grad :", layer.weights_grad[i][j], "\nexpect_grad :", expect_grad, "\nerror       :", layer.weights_grad[i][j] - expect_grad)
                print("----------------------------------------------")
                

#该程序用来检测梯度计算是否正确，从而可以判断神经网络代码实现是否正确
def test_grad_VECTOR():
    sample = np.array([0.9, 0.1, 0.1, 0.9, 0.8, 0.7, 0.6]).reshape(7,1)
    lable = np.array([0.9, 0.1, 0.1, 0.9, 0.8, 0.7, 0.6]).reshape(7,1)
    func = activate_function.ActivateFun()
    net = bbnet.Fully_Connected_Network_Vector([7,30,7],func)
    grad_check_vector(net, lable, sample)



def transpose(args):
    return list(map(
        lambda arg: list(map(
            lambda line: np.array(line).reshape(len(line), 1)
            , arg))
        , args
    ))
    
def train_MNIST_and_evaluate_VECTOR():
    last_error_ratio = 1.0
    epoch = 0
    print("reading datas.........")
    train_data_set, train_labels = transpose(get_training_data_set())
    test_data_set, test_labels = transpose(get_test_data_set())
    print("read datas finished!------")
    func = activate_function.ActivateFun()
    network = bbnet.Fully_Connected_Network_Vector([784, 300, 10], func)
    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.003, 1)
        print('%s epoch %d finished' % (datetime.datetime.now(), epoch))
        if epoch % 1 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print('%s after epoch %d, error ratio is %f' % (datetime.datetime.now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio
#-------------------------------------------------调用上述3个应用进行实现-------------------------------------------------------
# test_grad()
# test_grad_VECTOR()
# train_Y_X()
# train_Y_X_VECTOR()
# train_MNIST_and_evaluate()
train_MNIST_and_evaluate_VECTOR()