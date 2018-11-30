# coding=utf-8
# Author:Dodo
# Date:2018-11-30
# Email:lvtengchao@pku.edu.cn
# Blog:www.pkudodo.com

'''
数据集：Mnist
训练集数量：60000(实际使用:20000)
测试集数量：10000
------------------------------
运行结果：
    正确率：96.9%
    运行时长：8.8h

备注：对于mnist而言，李航的统计学习方法中有一些关键细节没有阐述，
建议先阅读我的个人博客，其中有详细阐述。阅读结束后再看该程序。
Blog：www.pkudodo.com
'''

import time
import numpy as np
from collections import defaultdict

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
        #十分类，list中放置标签
        if int(curLine[0]) == 0:
            labelList.append(1)
        else:
            labelList.append(0)
        #二值化
        dataList.append([int(int(num) > 128) for num in curLine[1:]])

    #返回data和label
    return dataList, labelList


class maxEnt:
    '''
    最大熵类
    '''
    def __init__(self, trainDataList, trainLabelList, testDataList, testLabelList):
        '''
        各参数初始化
        '''
        self.trainDataList = trainDataList          #训练数据集
        self.trainLabelList = trainLabelList        #训练标签集
        self.testDataList = testDataList            #测试数据集
        self.testLabelList = testLabelList          #测试标签集
        self.featureNum = len(trainDataList[0])     #特征数量

        self.N = len(trainDataList)                 #总训练集长度
        self.n = 0                                  #训练集中（xi，y）对数量
        self.M = 10000                              #
        self.fixy = self.calc_fixy()                #所有(x, y)对出现的次数
        self.w = [0] * self.n                       #Pw(y|x)中的w
        self.xy2idDict, self.id2xyDict = self.createSearchDict()        #(x, y)->id和id->(x, y)的搜索字典
        self.Ep_xy = self.calcEp_xy()               #Ep_xy期望值

    def calcEpxy(self):
        '''
        计算特征函数f(x, y)关于模型P(Y|X)与经验分布P_(X, Y)的期望值（P后带下划线“_”表示P上方的横线
        程序中部分下划线表示“|”，部分表示上方横线，请根据具体公式自行判断,）
        即“6.2.2 最大熵模型的定义”中第二个期望（83页最上方的期望）
        :return:
        '''
        #初始化期望存放列表，对于每一个xy对都有一个期望
        #这里的x是单个的特征，不是一个样本的全部特征。例如x={x1，x2，x3.....，xk}，实际上是（x1，y），（x2，y），。。。
        #但是在存放过程中需要将不同特诊的分开存放，李航的书可能是为了公式的泛化性高一点，所以没有对这部分提及
        #具体可以看我的博客，里面有详细介绍  www.pkudodo.com
        Epxy = [0] * self.n
        #对于每一个样本进行遍历
        for i in range(self.N):
            #初始化公式中的P(y|x)列表
            Pwxy = [0] * 2
            #计算P(y = 0 } X)
            #注：程序中X表示是一个样本的全部特征，x表示单个特征，这里是全部特征的一个样本
            Pwxy[0] = self.calcPwy_x(self.trainDataList[i], 0)
            #计算P(y = 1 } X)
            Pwxy[1] = self.calcPwy_x(self.trainDataList[i], 1)

            for feature in range(self.featureNum):
                for y in range(2):
                    if (self.trainDataList[i][feature], y) in self.fixy[feature]:
                        id = self.xy2idDict[feature][(self.trainDataList[i][feature], y)]
                        Epxy[id] += (1 / self.N) * Pwxy[y]
        return Epxy

    def calcEp_xy(self):
        '''
        计算特征函数f(x, y)关于经验分布P_(x, y)的期望值（下划线表示P上方的横线，
        同理Ep_xy中的“_”也表示p上方的横线）
        即“6.2.2 最大熵的定义”中第一个期望（82页最下方那个式子）
        :return: 计算得到的Ep_xy
        '''
        #初始化Ep_xy列表，长度为n
        Ep_xy = [0] * self.n

        #遍历每一个特征
        for feature in range(self.featureNum):
            #遍历每个特征中的(x, y)对
            for (x, y) in self.fixy[feature]:
                #获得其id
                id = self.xy2idDict[feature][(x, y)]
                #将计算得到的Ep_xy写入对应的位置中
                #fixy中存放所有对在训练集中出现过的次数，处于训练集总长度N就是概率了
                Ep_xy[id] = self.fixy[feature][(x, y)] / self.N

        #返回期望
        return Ep_xy

    def createSearchDict(self):
        '''
        创建查询字典
        xy2idDict：通过(x,y)对找到其id,所有出现过的xy对都有一个id
        id2xyDict：通过id找到对应的(x,y)对
        '''
        #设置xy搜多id字典
        #这里的x指的是单个的特征，而不是某个样本，因此将特征存入字典时也需要存入这是第几个特征
        #这一信息，这是为了后续的方便，否则会乱套。
        #比如说一个样本X = (0, 1, 1) label =(1)
        #生成的标签对有(0, 1), (1, 1), (1, 1)，三个(x，y)对并不能判断属于哪个特征的，后续就没法往下写
        #不可能通过(1, 1)就能找到对应的id，因为对于(1, 1),字典中有多重映射
        #所以在生成字典的时总共生成了特征数个字典，例如在mnist中样本有784维特征，所以生成784个字典，属于
        #不同特征的xy存入不同特征内的字典中，使其不会混淆
        xy2idDict = [{} for i in range(self.featureNum)]
        #初始化id到xy对的字典。因为id与(x，y)的指向是唯一的，所以可以使用一个字典
        id2xyDict = {}

        #设置缩影，其实就是最后的id
        index = 0
        #对特征进行遍历
        for feature in range(self.featureNum):
            #对出现过的每一个(x, y)对进行遍历
            #fixy：内部存放特征数目个字典，对于遍历的每一个特征，单独读取对应字典内的(x, y)对
            for (x, y) in self.fixy[feature]:
                #将该(x, y)对存入字典中，要注意存入时通过[feature]指定了存入哪个特征内部的字典
                #同时将index作为该对的id号
                xy2idDict[feature][(x, y)] = index
                #同时在id->xy字典中写入id号，val为(x, y)对
                id2xyDict[index] = (x, y)
                #id加一
                index += 1

        #返回创建的两个字典
        return xy2idDict, id2xyDict


    def calc_fixy(self):
        '''
        计算(x, y)在训练集中出现过的次数
        :return:
        '''
        #建立特征数目个字典，属于不同特征的(x, y)对存入不同的字典中，保证不被混淆
        fixyDict = [defaultdict(int) for i in range(self.featureNum)]
        #遍历训练集中所有样本
        for i in range(len(self.trainDataList)):
            #遍历样本中所有特征
            for j in range(self.featureNum):
                #将出现过的(x, y)对放入字典中并计数值加1
                fixyDict[j][(self.trainDataList[i][j], self.trainLabelList[i])] += 1
        #对整个大字典进行计数，判断去重后还有多少(x, y)对，写入n
        for i in fixyDict:
            self.n += len(i)
        #返回大字典
        return fixyDict


    def calcPwy_x(self, X, y):
        '''
        计算“6.23 最大熵模型的学习” 式6.22
        :param X: 要计算的样本X（一个包含全部特征的样本）
        :param y: 该样本的标签
        :return: 计算得到的Pw(Y|X)
        '''
        #分子
        numerator = 0
        #分母
        Z = 0
        #对每个特征进行遍历
        for i in range(self.featureNum):
            #如果该(xi,y)对在训练集中出现过
            if (X[i], y) in self.xy2idDict[i]:
                #在xy->id字典中指定当前特征i，以及(x, y)对：(X[i], y)，读取其id
                index = self.xy2idDict[i][(X[i], y)]
                #分子是wi和fi(x，y)的连乘再求和，最后指数
                #由于当(x, y)存在时fi(x，y)为1，因为xy对肯定存在，所以直接就是1
                #对于分子来说，就是n个wi累加，最后再指数就可以了
                #因为有n个w，所以通过id将w与xy绑定，前文的两个搜索字典中的id就是用在这里
                numerator += self.w[index]
            #同时计算其他一种标签y时候的分子，下面的z并不是全部的分母，再加上上式的分子以后
            #才是完整的分母，即z = z + numerator
            if (X[i], 1-y) in self.xy2idDict[i]:
                #原理与上式相同
                index = self.xy2idDict[i][(X[i], 1-y)]
                Z += self.w[index]
        #计算分子的指数
        numerator = np.exp(numerator)
        #计算分母的z
        Z = np.exp(Z) + numerator
        #返回Pw(y|x)
        return numerator / Z


    def maxEntropyTrain(self, iter = 500):
        #设置迭代次数寻找最优解
        for i in range(iter):
            #单次迭代起始时间点
            iterStart = time.time()

            #计算“6.2.3 最大熵模型的学习”中的第二个期望（83页最上方哪个）
            Epxy = self.calcEpxy()

            #使用的是IIS，所以设置sigma列表
            sigmaList = [0] * self.n
            #对于所有的n进行一次遍历
            for j in range(self.n):
                #依据“6.3.1 改进的迭代尺度法” 式6.34计算
                sigmaList[j] = (1 / self.M) * np.log(self.Ep_xy[j] / Epxy[j])

            #按照算法6.1步骤二中的（b）更新w
            self.w = [self.w[i] + sigmaList[i] for i in range(self.n)]

            #单次迭代结束
            iterEnd = time.time()
            #打印运行时长信息
            print('iter:%d:%d, time:%d'%(i, iter, iterStart - iterEnd))

    def predict(self, X):
        '''
        预测标签
        :param X:要预测的样本
        :return: 预测值
        '''
        #因为y只有0和1，所有建立两个长度的概率列表
        result = [0] * 2
        #循环计算两个概率
        for i in range(2):
            #计算样本x的标签为i的概率
            result[i] = self.calcPwy_x(X, i)
        #返回标签
        #max(result)：找到result中最大的那个概率值
        #result.index(max(result))：通过最大的那个概率值再找到其索引，索引是0就返回0，1就返回1
        return result.index(max(result))

    def test(self):
        '''
        对测试集进行测试
        :return:
        '''
        #错误值计数
        errorCnt = 0
        #对测试集中所有样本进行遍历
        for i in range(len(self.testDataList)):
            #预测该样本对应的标签
            result = self.predict(self.testDataList[i])
            #如果错误，计数值加1
            if result != self.testLabelList[i]:   errorCnt += 1
        #返回准确率
        return 1 - errorCnt / len(self.testDataList)

if __name__ == '__main__':
    start = time.time()

    # 获取训练集及标签
    print('start read transSet')
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')

    # 获取测试集及标签
    print('start read testSet')
    testData, testLabel = loadData('../Mnist/mnist_test.csv')

    #初始化最大熵类
    maxEnt = maxEnt(trainData[:20000], trainLabel[:20000], testData, testLabel)

    #开始训练
    print('start to train')
    maxEnt.maxEntropyTrain()

    #开始测试
    print('start to test')
    accuracy = maxEnt.test()
    print('the accuracy is:', accuracy)

    # 打印时间
    print('time span:', time.time() - start)