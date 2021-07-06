from functools import reduce
import random
import numpy as np

#-----------------------------------------------------Perception------------------------------------------------------------
#Single perception, using step function as the activate function
class Perceptioin(object):
    def __init__(self, input_numder, activate_fun):
        self.activate_fun = activate_fun
        self.weights = [0.0 for i in range(input_numder)]
        self.bias = 0.0

    def forward(self, input_vector):
        self.input_vector = input_vector
        return self.activate_fun(
            reduce(
                lambda a,b : a+b, list(map(
                    lambda a : a[0] * a[1], list(zip(input_vector, self.weights)))
                )
            , 0.0) + self.bias
        )
    def update(self, input_vector, output, label, learning_rate):
        error = label - output
        self.weights = list(map(
            lambda x: x[1] + learning_rate * error * x[0], list(zip(input_vector, self.weights))
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
        return "weights : %s\nbias : %s\n" % (self.weights, self.bias)

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

class Fully_Connected_Network(object):
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
                self.train_once(labels[j], samples[j], learning_rate)

#实现2 向量化编程,能够显著提高计算速度,几百倍
class Fully_Connection_Layer(object):
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

class Fully_Connected_Network_Vector(object):
    def __init__(self, layers, activate_function) -> None:
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Fully_Connection_Layer(layers[i], layers[i + 1], activate_function))

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
                self.train_once(lables[j], samples[j], learning_rate)

#-----------------------------------------------CNN network------------------------------------------------------
def padding(input, zp):
    if zp == 0:
        return input
    else:
        if input.ndim == 3:
            input_depth = input.shape[0]
            input_height = input.shape[1]
            input_width = input.shape[2]
            padding_array = np.zeros((input_depth, input_height + 2*zp, input_width + zp +zp))
            padding_array[:, zp : zp + input_height, zp : zp + input_width] = input
            return padding_array
        elif input.ndim == 2:
            input_height = input.shape[0]
            input_width = input.shape[1]
            padding_array = np.zeros(input_height + 2*zp, input_width + 2*zp)
            padding_array[zp : zp + input_height, zp + input_width] = input
            return padding_array

def get_patch(input, i, j, fliter_width, fliter_height, stride):
    start_i = i * stride
    start_j = j * stride
    if input.ndim == 2:
        return input[start_i : start_i + fliter_height, start_j : start_j + fliter_width]
    elif input.ndim == 3:
        return input[:, start_i : start_i + fliter_height, start_j : start_j + fliter_width]

def conv(input, weights, output, stride, bias):
    fliter_width = weights.shape[-1]
    fliter_height = weights.shape[-2]
    output_height = output.shape[0]
    output_width = output.shape[1]
    for i in range(output_height):
        for j in range(output_width):
            output[i][j] = (get_patch(input, i, j, fliter_width, fliter_height, stride) * weights).sum() + bias



class Fliter(object):
    def __init__(self, fliter_width, fliter_height, fliter_depth) -> None:
        self.weights = np.random.uniform(-1e-4, 1e-4, (fliter_depth, fliter_height, fliter_width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights = self.weights - learning_rate*self.weights_grad
        self.bias = self.bias - learning_rate*self.bias_grad

def element_wise_op(output, activate_function):
    for i in np.nditer(output, op_flags=["readwrite"]):
        i[...] = activate_function(i) 

class Convolution_Layer(object):
    def __init__(self, input_width, input_height, input_depth, fliter_width, fliter_height, fliter_number, zero_padding, stride, activate_function, learning_rate) -> None:
        self.input_height = input_height
        self.input_width = input_width
        self.channel_number = input_depth
        self.fliter_height = fliter_height
        self.fliter_width = fliter_width
        self.fliter_number = fliter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_heigth = self.calculate_output_size(self.input_height, self.fliter_height, self.zero_padding, self.stride)
        self.output_width = self.calculate_output_size(self.input_width, self.fliter_width, self.zero_padding, self.stride)
        self.output = np.zeros((self.fliter_number, self.output_heigth, self.output_width))
        self.fliters = []
        for i in range(self.fliter_number):
            self.fliters.append(Fliter(self.fliter_width, self.fliter_height, self.channel_number))
        self.activate_function = activate_function
        self.learning_rate = learning_rate

    def calculate_output_size(self, input_size, fliter_size, zero_padding, stride):
        return (input_size + 2*zero_padding -fliter_size) // stride + 1

    def forward(self, input):
        self.input = input
        self.padded_input = padding(self.input, self.zero_padding)
        for i in range(self.fliter_number):
            weights = self.fliters[i].get_weights()
            bias = self.fliters[i].get_bias()
            conv(self.padded_input, weights, self.output[i], self.stride, bias)
        element_wise_op(self.output, self.activate_function.relu_forward)

    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        expanded_width = (self.input_width - self.fliter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.fliter_height + 2 * self.zero_padding + 1)
        expand_array = np.zeros((depth, expanded_height, expanded_width))
        for i in range(self.output_heigth):
            for j in range(self.output_width):
                expand_array[:, i * self.stride, j * self.stride] = sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number,
            self.input_height, self.input_width))

    def bp_sensitivity(self, sensitivity_map, activate_function):
        expand_sensitivity_map = self.expand_sensitivity_map(sensitivity_map)
        expand_width = expand_sensitivity_map.shape[2]
        zp = (self.input_width + self.fliter_width -1 - expand_width) // 2
        padding_expand_sensitivity_map = padding(expand_sensitivity_map, zp)
        self.delta_array = self.create_delta_array()
        for f in range(len(self.fliters)):
            fliter = self.fliters[f]
            fliter_weights = np.array(list(map(lambda i : np.rot90(i, 2), fliter.get_weights())))
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padding_expand_sensitivity_map[f], fliter_weights, delta_array[d], 1, 0)
            self.delta_array += delta_array
        derivative_array = np.array(self.input)
        element_wise_op(derivative_array, activate_function.relu_backward)
        self.delta_array = self.delta_array * derivative_array

    def cacl_grad(self, sensitivity_map):
        expanded_array = self.expand_sensitivity_map(sensitivity_map)
        for f in range(self.fliter_number):
            # 计算每个权重的梯度
            fliter = self.fliters[f]
            for d in range(fliter.weights.shape[0]):
                conv(self.padded_input[d], 
                     expanded_array[f],
                     fliter.weights_grad[d], 1, 0)
            # 计算偏置项的梯度
            fliter.bias_grad = expanded_array[f].sum()

    def backward(self, sensitivity_map, activate_function):
        self.bp_sensitivity(sensitivity_map, activate_function)
        self.cacl_grad(sensitivity_map)

    def update(self):
        for flayer in self.fliters:
            flayer.update(self.learning_rate)

def get_max_index(array):
    max_i = 0
    max_j = 0
    max_value = array[0,0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] > max_value:
                max_value = array[i,j]
                max_i, max_j = i, j
    return max_i, max_j

class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height, 
                 channel_number, filter_width, 
                 filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = (input_width - 
            filter_width) // self.stride + 1
        self.output_height = (input_height -
            filter_height) // self.stride + 1
        self.output_array = np.zeros((self.channel_number,
            self.output_height, self.output_width))

    def forward(self, input_array):
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d,i,j] = (    
                        get_patch(input_array[d], i, j,
                            self.filter_width, 
                            self.filter_height, 
                            self.stride).max())

    def backward(self, input_array, sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        input_array[d], i, j,
                        self.filter_width, 
                        self.filter_height, 
                        self.stride)
                    k, l = get_max_index(patch_array)
                    self.delta_array[d, 
                        i * self.stride + k, 
                        j * self.stride + l] = \
                        sensitivity_array[d,i,j]