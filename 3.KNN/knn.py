#coding=utf-8
#Author:Dodo
#Date:2018-11-16
#Rebuild:2022-05-17
#Email:lvtengchao@pku.edu.cn



import numpy as np
import time

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
            # k近邻支持多分类，因此0-9共10个类别
            label.append(cur_label)
    assert len(data) == len(label)
    print('从{}导入样本数目:{}'.format(path, len(data)))
    return data, label



class KNN():
    def __init__(
            self,
            train_data,
            train_label,
            test_data,
            test_label,
    ):
        # 将数据转换成矩阵形式（在机器学习中因为通常都是向量的运算，转换称矩阵形式方便运算）
        # 转换后的数据尺寸为1000x784
        self.train_data = np.mat(train_data)
        # 将标签转换成矩阵，之后转置(.T为转置), 尺寸为784x1
        # 转置是因为在运算中需要单独取label中的某一个元素，如果是1xN的矩阵的话，无法用label[i]的方式读取
        self.train_label = np.mat(train_label).T
        self.test_data = np.mat(test_data)
        self.test_label = np.mat(test_label).T

    def calc_dist(self, x1, x2):
        '''
        计算两个样本点向量之间的距离
        使用的是欧氏距离，即 样本点每个元素相减的平方  再求和  再开方
        欧式举例公式这里不方便写，可以百度或谷歌欧式距离（也称欧几里得距离）
        :param x1:向量1
        :param x2:向量2
        :return:向量之间的欧式距离
        '''

        # x1 – x2为元素级别的相减，例如x1=[5, 6, 7], x2=[2, 2, 2]，则x1-x2=[3, 4, 5]
        # np.square()对应于元素的平方，结果仍然为一个向量
        # np.sum()为向量内所有元素和
        return np.sqrt(np.sum(np.square(x1 - x2)))

        # 马哈顿距离
        # return np.sum(np.fabs(x1 - x2))

        # 切比雪夫距离
        # return np.max(np.fabs(x1 - x2))

    def get_closest(self, x, topK):
        '''
        预测样本x的标记。
        获取方式通过找到与样本x最近的topK个点，并查看它们的标签。
        查找里面占某类标签最多的那类标签
        :param trainDataMat:训练集数据集
        :param trainLabelMat:训练集标签集
        :param x:要预测的样本x
        :param topK:选择参考最邻近样本的数目（样本数目的选择关系到正确率，详看3.2.3 K值的选择）
        :return:预测的标记
        '''
        # 建立一个存放向量x与每个训练集中样本距离的列表
        # 列表的长度为训练集的长度，distList[i]表示x与训练集中第
        ## i个样本的距离
        distList = [0] * self.train_data.shape[0]
        # 遍历训练集中所有的样本点，计算与x的距离
        for i in range(self.train_data.shape[0]):
            # 获取训练集中当前样本的向量
            x1 =self.train_data[i]
            # 计算向量x与训练集样本x的距离
            curDist = self.calc_dist(x1, x)
            # 将距离放入对应的列表位置中
            distList[i] = curDist

        # 对距离列表进行排序
        # argsort：函数将数组的值从小到大排序后，并按照其相对应的索引值输出
        # 例如：
        #   >>> x = np.array([3, 1, 2])
        #   >>> np.argsort(x)
        #   array([1, 2, 0])
        # 返回的是列表中从小到大的元素索引值，对于我们这种需要查找最小距离的情况来说很合适
        # array返回的是整个索引值列表，我们通过[:topK]取列表中前topL个放入list中。
        # ----------------优化点-------------------
        # 由于我们只取topK小的元素索引值，所以其实不需要对整个列表进行排序，而argsort是对整个
        # 列表进行排序的，存在时间上的浪费。字典有现成的方法可以只排序top大或top小，可以自行查阅
        # 对代码进行稍稍修改即可
        # 这里没有对其进行优化主要原因是KNN的时间耗费大头在计算向量与向量之间的距离上，由于向量高维
        # 所以计算时间需要很长，所以如果要提升时间，在这里优化的意义不大。（当然不是说就可以不优化了，
        # 主要是我太懒了）
        topKList = np.argsort(np.array(distList))[:topK]  # 升序排序
        # 建立一个长度时的列表，用于选择数量最多的标记
        # 3.2.4提到了分类决策使用的是投票表决，topK个标记每人有一票，在数组中每个标记代表的位置中投入
        # 自己对应的地方，随后进行唱票选择最高票的标记
        labelList = [0] * 10
        # 对topK个索引进行遍历
        for index in topKList:
            # trainLabelMat[index]：在训练集标签中寻找topK元素索引对应的标记
            # int(trainLabelMat[index])：将标记转换为int（实际上已经是int了，但是不int的话，报错）
            # labelList[int(trainLabelMat[index])]：找到标记在labelList中对应的位置
            # 最后加1，表示投了一票
            labelList[int(self.train_label[index])] += 1
        # max(labelList)：找到选票箱中票数最多的票数值
        # labelList.index(max(labelList))：再根据最大值在列表中找到该值对应的索引，等同于预测的标记
        return labelList.index(max(labelList))

    def eval(self, topK=10):
        # 错误值计数
        errorCnt = 0
        # 遍历测试集，对每个测试集样本进行测试
        for i in range(self.test_data.shape[0]):
            # 读取测试集当前测试样本的向量
            x = self.test_data[i]
            # 找最近的topK个点，进行类别投票得到预测
            y = self.get_closest(x, topK)
            # 如果预测标记与实际标签不符，错误值计数加1
            if y != self.test_label[i]: errorCnt += 1

        # 返回正确率
        # return 1 - (errorCnt / len(testDataMat))
        return 1 - (errorCnt / self.test_data.shape[0])

if __name__ == "__main__":
    # 记录开始时间
    start = time.time()
    # 读取数据集
    train_data, train_label = load_data('../0.datasets/Mnist/Mnist_train.txt')
    test_data, test_label = load_data('../0.datasets/Mnist/Mnist_test.txt')
    # 初始化感知机类
    knn = KNN(
        train_data=train_data,
        train_label=train_label,
        test_data=test_data,
        test_label=test_label,
    )

    # KNN没有显式的训练过程，因此可以直接进行推理评估
    # 评估，计算准确率
    accuracy = knn.eval()
    # 获取结束时间
    end = time.time()
    # 显示用时时长
    print('用时时长: {} s'.format(end - start))
    # 显示准确率
    print('准确率: {}%'.format(accuracy * 100))
