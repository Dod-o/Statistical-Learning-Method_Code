# coding=utf-8
# Author:Dodo
# Date:2018-11-27
# Email:lvtengchao@pku.edu.cn
# Blog:www.pkudodo.com

'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
    正确率：98.91%
    运行时长：59s
'''

import time
import numpy as np


def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    # 存放数据及标记的list
    dataList = []; labelList = []
    # 打开文件
    fr = open(fileName, 'r')
    # 将文件按行读取
    for line in fr.readlines():
        # 对每一行数据按切割福','进行切割，返回字段列表
        curLine = line.strip().split(',')

        # Mnsit有0-9是个标记，由于是二分类任务，所以将标记0的作为1，其余为0
        # 验证过<5为1 >5为0时正确率在90%左右，猜测是因为数多了以后，可能不同数的特征较乱，不能有效地计算出一个合理的超平面
        # 查看了一下之前感知机的结果，以5为分界时正确率81，重新修改为0和其余数时正确率98.91%
        # 看来如果样本标签比较杂的话，对于是否能有效地划分超平面确实存在很大影响
        if int(curLine[0]) == 0:
            labelList.append(1)
        else:
            labelList.append(0)
        #存放标记
        #[int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一哥元素（标记）外将所有元素转换成int类型
        #[int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)
        dataList.append([int(num)/255 for num in curLine[1:]])
        # dataList.append([int(num) for num in curLine[1:]])

    #返回data和label
    return dataList, labelList

def predict(w, x):
    '''
    预测标签
    :param w:训练过程中学到的w
    :param x: 要预测的样本
    :return: 预测结果
    '''
    #dot为两个向量的点积操作，计算得到w * x
    wx = np.dot(w, x)
    #计算标签为1的概率
    #该公式参考“6.1.2 二项逻辑斯蒂回归模型”中的式6.5
    P1 = np.exp(wx) / (1 + np.exp(wx))
    #如果为1的概率大于0.5，返回1
    if P1 >= 0.5:
        return 1
    #否则返回0
    return 0

def logisticRegression(trainDataList, trainLabelList, iter = 200):
    '''
    逻辑斯蒂回归训练过程
    :param trainDataList:训练集
    :param trainLabelList: 标签集
    :param iter: 迭代次数
    :return: 习得的w
    '''
    #按照书本“6.1.2 二项逻辑斯蒂回归模型”中式6.5的规则，将w与b合在一起，
    #此时x也需要添加一维，数值为1
    #循环遍历每一个样本，并在其最后添加一个1
    for i in range(len(trainDataList)):
        trainDataList[i].append(1)

    #将数据集由列表转换为数组形式，主要是后期涉及到向量的运算，统一转换成数组形式比较方便
    trainDataList = np.array(trainDataList)
    #初始化w，维数为样本x维数+1，+1的那一位是b，初始为0
    w = np.zeros(trainDataList.shape[1])

    #设置步长
    h = 0.001

    #迭代iter次进行随机梯度下降
    for i in range(iter):
        #每次迭代冲遍历一次所有样本，进行随机梯度下降
        for j in range(trainDataList.shape[0]):
            #随机梯度上升部分
            #在“6.1.3 模型参数估计”一章中给出了似然函数，我们需要极大化似然函数
            #但是似然函数由于有求和项，并不能直接对w求导得出最优w，所以针对似然函数求和
            #部分中每一项进行单独地求导w，得到针对该样本的梯度，并进行梯度上升（因为是
            #要求似然函数的极大值，所以是梯度上升，如果是极小值就梯度下降。梯度上升是
            #加号，下降是减号）
            #求和式中每一项单独对w求导结果为：xi * yi - (exp(w * xi) * xi) / (1 + exp(w * xi))
            #如果对于该求导式有疑问可查看我的博客 www.pkudodo.com

            #计算w * xi，因为后式中要计算两次该值，为了节约时间这里提前算出
            #其实也可直接算出exp(wx)，为了读者能看得方便一点就这么写了，包括yi和xi都提前列出了
            wx = np.dot(w, trainDataList[j])
            yi = trainLabelList[j]
            xi = trainDataList[j]
            #梯度上升
            w +=  h * (xi * yi - (np.exp(wx) * xi) / ( 1 + np.exp(wx)))

    #返回学到的w
    return w

def model_test(testDataList, testLabelList, w):
    '''
    验证
    :param testDataList:测试集
    :param testLabelList: 测试集标签
    :param w: 训练过程中学到的w
    :return: 正确率
    '''

    #与训练过程一致，先将所有的样本添加一维，值为1，理由请查看训练函数
    for i in range(len(testDataList)):
        testDataList[i].append(1)

    #错误值计数
    errorCnt = 0
    #对于测试集中每一个测试样本进行验证
    for i in range(len(testDataList)):
        #如果标记与预测不一致，错误值加1
        if testLabelList[i] != predict(w, testDataList[i]):
            errorCnt += 1
    #返回准确率
    return 1 - errorCnt / len(testDataList)



if __name__ == '__main__':
    start = time.time()

    # 获取训练集及标签
    print('start read transSet')
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')

    # 获取测试集及标签
    print('start read testSet')
    testData, testLabel = loadData('../Mnist/mnist_test.csv')

    # 开始训练，学习w
    print('start to train')
    w = logisticRegression(trainData, trainLabel)

    #验证正确率
    print('start to test')
    accuracy = model_test(testData, testLabel, w)

    # 打印准确率
    print('the accuracy is:', accuracy)
    # 打印时间
    print('time span:', time.time() - start)

