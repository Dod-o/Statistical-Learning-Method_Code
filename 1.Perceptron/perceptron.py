#coding=utf-8
#Author:Dodo
#Date:2018-11-15
#Rebuild:2022-03-21
#Email:lvtengchao@pku.edu.cn

import time
import numpy as np

def load_data(path):
    '''
    加载Mnist数据集
    Args:
        path (string):
            数据集路径
    Return:
        data (list[list[int]]):
            样本数据，list形式，每个元素为一个list，描述一个样本
        label (list[int]):
            样本对应的标签
    '''
    data, label = [], []
    with open(path, 'r', encoding='utf-8') as fr:
        # 按行读取文件内容
        for line in fr.readlines():
            # strip()操作删除每行前后的回车符，在获取文本行数据时通常会添加改操作
            # split(',')将行按分隔符","切割，转换成列表
            fields = line.strip().split(',')
            # 读取到的文本均为字符形式，使用map函数将其转换为int类型
            fields = list(map(int, fields))
            # 行结构为 label+data，data为784长度，所以总长度为1+784=785
            assert len(fields) == 785
            # 得到样本数据和对应标签
            cur_label, cur_data = fields[0], fields[1:]
            data.append(cur_data)
            # 感知机为二分类器，但mnist包含0～9共10种标签，所以我们将小于5的数分为1类，大于5的数分为1类，进行二分类任务。
            if cur_label < 5:
                label.append(-1)
            else:
                label.append(1)
    assert len(data) == len(label)
    # print('从{}导入样本数目:{}'.format(path, len(data)))
    print('load {} samples from {}'.format(len(data), path))
    return data, label

class Perceptron():
    def __init__(
            self,
            train_data,
            train_label,
            test_data,
            test_label,
            h=0.0001,
            epoch=30,
    ):
        # 将数据转换成矩阵形式（在机器学习中因为通常都是向量的运算，转换称矩阵形式方便运算）
        # 转换后的数据尺寸为1000x784
        self.train_data = np.mat(train_data)
        # 将标签转换成矩阵，之后转置(.T为转置), 尺寸为784x1
        # 转置是因为在运算中需要单独取label中的某一个元素，如果是1xN的矩阵的话，无法用label[i]的方式读取
        self.train_label = np.mat(train_label).T
        self.test_data = np.mat(test_data)
        self.test_label = np.mat(test_label).T

        # 创建初始权重w，初始值全为0。
        # 当data的尺寸是MxN时，data.shape的返回值为M，N -> data.shape[1]的值即为N，与
        # 样本维度保持一致
        self.w = np.zeros((1, self.train_data.shape[1]))
        # 初始化偏置b为0
        self.b = 0
        # 初始化梯度下降过程中的步长，控制梯度下降速率
        self.h = h
        # 训练的轮数，一轮表示训练集中所有数据参与一次训练
        self.epoch = epoch

    def train(self):
        print('start training...')
        m, n = self.train_data.shape

        for epoch in range(self.epoch):
            # 打印训练进度
            print('training: epoch-{}'.format(epoch))

            for i in range(m):
                # 获取当前样本的向量
                xi = self.train_data[i]
                # 获取当前样本所对应的标签
                yi = self.train_label[i]
                # 判断是否是误分类样本
                # 误分类样本特诊为： -yi(w*xi+b)>=0，详细可参考书中2.2.2小节           @@@@@@@@@@@@@@@
                # 在书的公式中写的是>0，实际上如果=0，说明该点在超平面上，也是不正确的
                if -1 * yi * (self.w * xi.T + self.b) >= 0:
                    # 对于误分类样本，进行梯度下降，更新w和b
                    self.w = self.w + self.h * yi * xi
                    self.b = self.b + self.h * yi

    def eval(self):
        # 错误样本数计数
        error_cnt = 0
        m, n = self.test_data.shape
        # 遍历所有测试样本
        for i in range(m):
            # 获得单个样本向量
            xi = self.test_data[i]
            # 获得该样本标记
            yi = self.test_label[i]
            # 获得运算结果
            result = -1 * yi * (self.w * xi.T + self.b)
            # 如果-yi(w*xi+b)>=0，说明该样本被误分类，错误样本数加一
            if result >= 0: error_cnt += 1
        # 正确率 = 1 - （样本分类错误数 / 样本总数）
        accuracy = 1 - (error_cnt / m)
        return accuracy

if __name__ == '__main__':
    # 记录开始时间
    start = time.time()
    # 读取数据集
    train_data, train_label = load_data('../0.datasets/Mnist/Mnist_train.txt')
    test_data, test_label = load_data('../0.datasets/Mnist/Mnist_test.txt')
    # 初始化感知机类
    perceptron = Perceptron(
        train_data=train_data,
        train_label=train_label,
        test_data=test_data,
        test_label=test_label,
    )
    # 训练
    perceptron.train()
    # 评估，计算准确率
    accuracy = perceptron.eval()
    # 获取结束时间
    end = time.time()
    # 显示用时时长
    print('time span: {} s'.format(end - start))
    # 显示准确率
    print('accuracy is: {}%'.format(accuracy * 100))
