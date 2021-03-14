from functools import reduce

#------------------------------------------用于感知机的异或数据集---------------------------------------
class xor_data(object):
    def __init__(self):
        self.samples = [[1,1], [0,0], [1,0], [0,1]]
        self.labels = [1, 0, 0, 0]

#---------------------------------------用于线性单元的工资数据集----------------------------------------
class liner_unit_data(object):
    def __init__(self):
        self.input_vecs = [[5], [3], [8], [1.4], [10.1]]
        self.labels = [5500, 2300, 7600, 1800, 11400]

#---------------------------------------用于全连接训练测试1的数据集-------------------------------------
class Normalizer(object):
    def __init__(self,) -> None:
        self.mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]

    def normal(self, data):
        return list(map(
            lambda m : 0.9 if m & data else 0.1, self.mask
        ))

    def denormal(self, vector):
        binary = list(map(
            lambda m : 1 if m > 0.5 else 0, vector
        ))
        for i in range(len(binary)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(
            lambda a, b : a+ b, binary, 0
        )

#-----------------------------------------用于MINIST数据集读取----------------------------------------
class Loader(object):
    def __init__(self, path, count):
        '''
        初始化加载器
        path: 数据文件路径
        count: 文件中的样本个数
        '''
        self.path = path
        self.count = count
        
    def get_file_content(self):
        '''
        读取文件内容
        '''
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content

# 图像数据加载器
class ImageLoader(Loader):
    def get_picture(self, content, index):
        '''
        内部函数，从文件中获取图像
        '''
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(content[start + i * 28 + j])
        return picture

    def get_one_sample(self, picture):
        '''
        内部函数，将图像转化为样本的输入向量
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        '''
        加载数据文件，获得全部样本的输入向量
        '''
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)))
        return data_set

# 标签数据加载器
class LabelLoader(Loader):
    def load(self):
        '''
        加载数据文件，获得全部样本的标签向量
        '''
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels

    def norm(self, label):
        '''
        内部函数，将一个值转换为10维标签向量
        '''
        label_vec = []
        label_value = label
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec

#-----------------------------------------数据集----------------------------------------