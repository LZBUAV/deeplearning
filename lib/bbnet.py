from functools import reduce
from operator import le
from os import error
from lib import activate_function
import random
import numpy as np

#-----------------------------------------------------Perception------------------------------------------------------------
#Single perception, using step function as the activate function
class Perceptioin(object):
    def __init__(self, input_numder, activate_fun):
        self.activate_fun = activate_fun
        self.weigths = [0.0 for i in range(input_numder)]
        self.bias = 0.0

    def forward(self, input_vector):
        self.input_vector = input_vector
        return self.activate_fun(
            reduce(
                lambda a,b : a+b, list(map(
                    lambda a : a[0] * a[1], list(zip(input_vector, self.weigths)))
                )
            , 0.0) + self.bias
        )
    def update(self, input_vector, output, label, learning_rate):
        error = label - output
        self.weigths = list(map(
            lambda x: x[1] + learning_rate * error * x[0], list(zip(input_vector, self.weigths))
        ))
        self.bias = self.bias + learning_rate * error

    def train_once(self, input_vectors, labels, learning_rate):
        samples = list(zip(input_vectors, labels))
        for sample in samples:
            output = self.forward(sample[0])
            self.update(sample[0], output, sample[1], learning_rate)

    def train(self, input_vectors, labels, learning_rate, epoch):
        for i in range(epoch):
            self.train_once(input_vectors, labels, learning_rate)
            print("epoch-----", i + 1)

    def __str__(self) -> str:
        return "weigths : %s\nbias : %s\n" % (self.weigths, self.bias)

#-----------------------------------------------------Liner Unit------------------------------------------------------------
#single liner unit, using the liner function as the activate function
class LinerUnit(Perceptioin):
    def __init__(self, input_numder, activate_fun):
        Perceptioin.__init__(self, input_numder, activate_fun)

#---------------------------------------Full connection network / multilayer perception-------------------------------------
#Full connection network / multilayer perception, using sigmod function as the activate function
#实现1 面向对象编程
class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.grad = 0.0

    def cal_grad(self):
        self.grad = self.downstream_node.delta * self.upstream_node.output

    def update_weight(self, learning_rate):
        self.weight = self.weight + learning_rate * self.grad

    def get_grad(self):
        return self.grad

class Node(object):
    def __init__(self, layer_index, node_index, activate_function):
        self.layer_inder = layer_index
        self.node_index = node_index
        self.activate_function = activate_function
        self.upstream_nodes = []
        self.downstream_nodes = []
        self.output = 0.0
        self.delta = 0.0

    def set_output(self, output_value):
        self.output = output_value

    def append_upstream_node(self, conn):
        self.upstream_nodes.append(conn)

    def append_downstream_node(self, conn):
        self.downstream_nodes.append(conn)

    def forward(self):
        weighted_input = reduce(lambda a ,b : a+ b,
            list(map(
                lambda conn : conn.weight * conn.upstream_node.output, self.upstream_nodes
            )), 0.0
        )
        self.output = self.activate_function(weighted_input)

    def cal_hidden_layer_delta(self):
        detla_from_next_layer = reduce(lambda  a ,b : a + b,
            list(map(
                lambda conn : conn.downstream_node.delta * conn.weight, self.downstream_nodes
            )), 0.0
        )
        self.delta = detla_from_next_layer * self.output * (1 - self.output)

    def cal_output_layer_delta(self, label):
        error = label - self.output
        self.delta = error * self.output * (1 - self.output)

class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.output = 1
        self.delta = 0.0
        self.downstream_nodes = []

    def append_downstream_node(self, conn):
        self.downstream_nodes.append(conn)

    def cal_hidden_layer_delta(self):
        detla_from_next_layer = reduce(lambda  a ,b : a + b,
            list(map(
                lambda conn : conn.downstream_node.delta * conn.weight, self.downstream_nodes
            )), 0.0
        )
        self.delta = detla_from_next_layer * self.output * (1 - self.output)

class Layer(object):
    def __init__(self, layer_index, node_count, activate_function) -> None:
        self.layer_index = layer_index
        self.node_count = node_count
        self.activate_function = activate_function
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i, activate_function))
        self.nodes.append(ConstNode(layer_index, node_count))
    
    def set_output(self, output_datas):
        for i in range(len(output_datas)):
            self.nodes[i].set_output(output_datas[i])

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.forward()

class Connections(object):
    def __init__(self) -> None:
        self.connections = []
    
    def add_connectioin(self, conn):
        self.connections.append(conn)

class Network(object):
    def __init__(self, layers, activate_function) -> None:
        self.layers = []
        self.activate_function = activate_function
        self.connections = Connections()

        for i in range(len(layers)):
            self.layers.append(Layer(i, layers[i], activate_function))

        for i in range(len(layers) - 1):
            for upstream_node in self.layers[i].nodes:
                for downstream_node in self.layers[i+1].nodes[:-1]:
                    connection = Connection(upstream_node, downstream_node)
                    connection.upstream_node.append_downstream_node(connection)
                    connection.downstream_node.append_upstream_node(connection)
                    self.connections.add_connectioin(connection)

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for layer in self.layers:
            layer.calc_output()
        output =  list(map(
            lambda node : node.output, self.layers[-1].nodes[:-1]
            ))
        return output

    def calc_delta(self, label):
        for i in range(len(label)):
            self.layers[-1].nodes[i].cal_output_layer_delta(label[i])

        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.cal_hidden_layer_delta()

    def cal_grad(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream_nodes:
                    conn.cal_grad()
    
    def get_grad(self, lable, sample):
        self.predict(sample)
        self.calc_delta(lable)
        self.cal_grad()

    def update(self, learning_rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream_nodes:
                    conn.update_weight(learning_rate)

    def train_once(self, lable, sample, learning_rate):
        output = self.predict(sample)
        self.calc_delta(lable)
        self.cal_grad()
        self.update(learning_rate)

        loss = 0.5 * reduce(
        lambda a, b : a + b, list(
            map(
                lambda a : (a[0] - a[1]) * (a[0] - a[1]), list(zip(lable, output))
            )
        ), 0.0
    )
        return loss

    def train(self, labels, samples, learning_rate, epoch):
        for i in range(epoch):
            for j in range(len(samples)):
                loss = self.train_once(labels[j], samples[j], learning_rate)
                if j % 30  == 0:
                    print("epoch :", i + 1, "|  sample", j + 1, "|  loss :", loss)
            print("epoch :", i + 1, "finished!")

#实现2 向量化编程,能够显著提高计算速度,几百倍
class FullyConnect(object):
    def __init__(self, input_size, output_size, activate_function):
        self.input_size = input_size
        self.output_size = output_size
        self.activate_function = activate_function
        self.output = np.zeros((output_size, 1))
        self.bias = np.zeros((output_size, 1))
        self.weights = np.random.uniform(-0.1, 0.1, (output_size, input_size))

    def forward(self, sample):
        self.input = sample
        self.output = self.activate_function.sigmod_forward(np.dot(self.weights, self.input) + self.bias)

    def backword(self, delta):
        self.delta = self.activate_function.sigmod_backward(self.input) * np.dot(self.weights.T, delta)
        self.weights_grad = np.dot(delta, self.input.T)
        self.bias_grad = delta 

    def update(self, learning_rate):
        self.weights = self.weights + learning_rate* self.weights_grad
        self.bias = self.bias + learning_rate * self.bias_grad

class Network_Vector(object):
    def __init__(self, layers, activate_function) -> None:
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullyConnect(layers[i], layers[i + 1], activate_function))

    def predict(self, sample):
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def calc_delta(self, lable):
        delta = (lable - self.layers[-1].output) * self.layers[-1].activate_function.sigmod_backward(self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backword(delta)
            delta = layer.delta
        return delta

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def train_once(self, lable, sample, learning_rate):
        output = self.predict(sample)
        self.calc_delta(lable)
        self.update(learning_rate)
        loss = 0.5 * ((lable - output) * (lable - output)).sum()
        return loss

    def train(self, lables, samples, learning_rate, epoch):
        for i in range(epoch):
            for j in range(len(samples)):
                loss = self.train_once(lables[j], samples[j], learning_rate)
            print("epoch :", i + 1, "|  loss :", loss)