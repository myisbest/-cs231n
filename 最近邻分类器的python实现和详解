# coding=utf-8
'''
假设我们获得了50,000张图像的CIFAR-10训练集（每个标签有5000张图像），我们希望标注剩余的10,000张图像。最近的邻居分类器将选取一张测试图像，
将其与每一个训练图像进行比较，在训练集中找到最接近的单个图像,然后将这个测试图像归类到这个最接近图像的类别。
'''
# python的pickle模块实现了基本的数据序列和反序列化。通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，永久存储；
# 通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象。
import pickle  
import os
import sys
import numpy as np


def load_CIFAR_batch(filename):
    """
    cifar-10数据集是分batch存储的，这是载入单个batch
    @参数 filename: cifar文件名
    @r返回值: X, Y: cifar batch中的 data 和 labels
    """

    with open(filename, 'rb') as  fo:
        datadict = pickle.load(fo, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        # 数组重构函数reshape: 3表示三维空间RGB，reshape函数把X这个二维列表变成10000个32*32的矩阵，每个矩阵的元素是具有3个元素的一维列表
        # reshape函数把X这个二维列表变成了四维列表。
        # transpose 作用是改变序列，本来四维列表的坐标轴顺序是0,1,2,3，transpose把它变成了0,2,3,1，transpose是为了方便之后的tensorflow 模型使用。
        # 通常我们读入的数据格式是（32，32，3）即通道数在最后。像keras的输入格式就是（3，32，32）这样的
        # astype：改变整个X中的数据类型为float
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        # 把Y变成numpy数组类型
        Y = np.array(Y)

        return X, Y


def load_CIFAR10(ROOT):
    """
    读取载入整个 CIFAR-10 数据集
    @参数 ROOT: 根目录名
    @return: X_train, Y_train: 训练集 data 和 labels
             X_test, Y_test: 测试集 data 和 labels
    """

    xs = []
    ys = []

    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        # xs.append(X)将5个batch整合起来；np.concatenate(xs)使得最终X_train的尺寸为(50000,32,32,3)
        xs.append(X)
        ys.append(Y)

    X_train = np.concatenate(xs)  # 使变xs成行向量
    Y_train = np.concatenate(ys)

    del X, Y

    X_test, Y_test = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))

    return X_train, Y_train, X_test, Y_test


# 载入训练和测试数据集
X_train, Y_train, X_test, Y_test = load_CIFAR10('F:/PyCharmProject/Deep learning/deep learning/cifar-10-batches-py/')
# 把32*32*3的多维数组展平，把所有的图像都拉伸成行
Xtr_rows = X_train.reshape(X_train.shape[0], 32 * 32 * 3)  # Xtr_rows : 50000 x 3072
Xte_rows = X_test.reshape(X_test.shape[0], 32 * 32 * 3)  # Xte_rows : 10000 x 3072

'''
作为评估标准，通常使用精确度来度量预测的正确部分。train(X,y)实现传入训练数据X和对应标签y并从中学习的功能。
然后是一个predict(X)函数，它接收新的数据并预测其标签。
'''
class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """
        这个地方的训练其实就是把所有的已有图片读取进来 -_-||
        """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """
        所谓的预测过程其实就是扫描所有训练集中的图片，计算距离，取最小的距离对应图片的类目
        """
        num_test = X.shape[0]   # 读取X的第一维的长度，这里测试集是10000张图片，所以num_test为10000
        # 要保证维度一致哦
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # 把训练集扫一遍 -_-||
        for i in range(num_test):
            # 计算l1距离，并找到最近的图片
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)  # 取最近图片的下标
            Ypred[i] = self.ytr[min_index]  # 记录下label

        return Ypred


nn = NearestNeighbor()  # 初始化一个最近邻对象
nn.train(Xtr_rows, Y_train)  # 训练...其实就是读取训练集
Yte_predict = nn.predict(Xte_rows)  # 预测
# 比对标准答案，计算准确率
print('accuracy: %f' % (np.mean(Yte_predict == Y_test)))
