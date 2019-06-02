# coding=utf-8
# Author:Dodo
# Date:2018-12-8
# Email:lvtengchao@pku.edu.cn
# Blog:www.pkudodo.com

'''
数据集：伪造数据集（两个高斯分布混合）
数据集长度：1000
------------------------------
运行结果：
----------------------------
the Parameters set is:
alpha0:0.3, mu0:0.7, sigmod0:-2.0, alpha1:0.5, mu1:0.5, sigmod1:1.0
----------------------------
the Parameters predict is:
alpha0:0.4, mu0:0.6, sigmod0:-1.7, alpha1:0.7, mu1:0.7, sigmod1:0.9
----------------------------
'''

import numpy as np
import random
import math
import time

def loadData(mu0, sigma0, mu1, sigma1, alpha0, alpha1):
    '''
    初始化数据集
    这里通过服从高斯分布的随机函数来伪造数据集
    :param mu0: 高斯0的均值
    :param sigma0: 高斯0的方差
    :param mu1: 高斯1的均值
    :param sigma1: 高斯1的方差
    :param alpha0: 高斯0的系数
    :param alpha1: 高斯1的系数
    :return: 混合了两个高斯分布的数据
    '''
    #定义数据集长度为1000
    length = 1000

    #初始化第一个高斯分布，生成数据，数据长度为length * alpha系数，以此来
    #满足alpha的作用
    data0 = np.random.normal(mu0, sigma0, int(length * alpha0))
    #第二个高斯分布的数据
    data1 = np.random.normal(mu1, sigma1, int(length * alpha1))

    #初始化总数据集
    #两个高斯分布的数据混合后会放在该数据集中返回
    dataSet = []
    #将第一个数据集的内容添加进去
    dataSet.extend(data0)
    #添加第二个数据集的数据
    dataSet.extend(data1)
    #对总的数据集进行打乱（其实不打乱也没事，只不过打乱一下直观上让人感觉已经混合了
    # 读者可以将下面这句话屏蔽以后看看效果是否有差别）
    random.shuffle(dataSet)

    #返回伪造好的数据集
    return dataSet

def calcGauss(dataSetArr, mu, sigmod):
    '''
    根据高斯密度函数计算值
    依据：“9.3.1 高斯混合模型” 式9.25
    注：在公式中y是一个实数，但是在EM算法中(见算法9.2的E步)，需要对每个j
    都求一次yjk，在本实例中有1000个可观测数据，因此需要计算1000次。考虑到
    在E步时进行1000次高斯计算，程序上比较不简洁，因此这里的y是向量，在numpy
    的exp中如果exp内部值为向量，则对向量中每个值进行exp，输出仍是向量的形式。
    所以使用向量的形式1次计算即可将所有计算结果得出，程序上较为简洁
    :param dataSetArr: 可观测数据集
    :param mu: 均值
    :param sigmod: 方差
    :return: 整个可观测数据集的高斯分布密度（向量形式）
    '''
    #计算过程就是依据式9.25写的，没有别的花样
    result = (1 / (math.sqrt(2 * math.pi) * sigmod**2)) * \
             np.exp(-1 * (dataSetArr - mu) * (dataSetArr - mu) / (2 * sigmod**2))
    #返回结果
    return result


def E_step(dataSetArr, alpha0, mu0, sigmod0, alpha1, mu1, sigmod1):
    '''
    EM算法中的E步
    依据当前模型参数，计算分模型k对观数据y的响应度
    :param dataSetArr: 可观测数据y
    :param alpha0: 高斯模型0的系数
    :param mu0: 高斯模型0的均值
    :param sigmod0: 高斯模型0的方差
    :param alpha1: 高斯模型1的系数
    :param mu1: 高斯模型1的均值
    :param sigmod1: 高斯模型1的方差
    :return: 两个模型各自的响应度
    '''
    #计算y0的响应度
    #先计算模型0的响应度的分子
    gamma0 = alpha0 * calcGauss(dataSetArr, mu0, sigmod0)
    #模型1响应度的分子
    gamma1 = alpha1 * calcGauss(dataSetArr, mu1, sigmod1)

    #两者相加为E步中的分布
    sum = gamma0 + gamma1
    #各自相除，得到两个模型的响应度
    gamma0 = gamma0 / sum
    gamma1 = gamma1 / sum

    #返回两个模型响应度
    return gamma0, gamma1

def M_step(muo, mu1, gamma0, gamma1, dataSetArr):
    #依据算法9.2计算各个值
    #这里没什么花样，对照书本公式看看这里就好了
    mu0_new = np.dot(gamma0, dataSetArr) / np.sum(gamma0)
    mu1_new = np.dot(gamma1, dataSetArr) / np.sum(gamma1)

    sigmod0_new = math.sqrt(np.dot(gamma0, (dataSetArr - muo)**2) / np.sum(gamma0))
    sigmod1_new = math.sqrt(np.dot(gamma1, (dataSetArr - mu1)**2) / np.sum(gamma1))

    alpha0_new = np.sum(gamma0) / len(gamma0)
    alpha1_new = np.sum(gamma1) / len(gamma1)

    #将更新的值返回
    return mu0_new, mu1_new, sigmod0_new, sigmod1_new, alpha0_new, alpha1_new


def EM_Train(dataSetList, iter = 500):
    '''
    根据EM算法进行参数估计
    算法依据“9.3.2 高斯混合模型参数估计的EM算法” 算法9.2
    :param dataSetList:数据集（可观测数据）
    :param iter: 迭代次数
    :return: 估计的参数
    '''
    #将可观测数据y转换为数组形式，主要是为了方便后续运算
    dataSetArr = np.array(dataSetList)

    #步骤1：对参数取初值，开始迭代
    alpha0 = 0.5; mu0 = 0; sigmod0 = 1
    alpha1 = 0.5; mu1 = 1; sigmod1 = 1

    #开始迭代
    step = 0
    while (step < iter):
        #每次进入一次迭代后迭代次数加1
        step += 1
        #步骤2：E步：依据当前模型参数，计算分模型k对观测数据y的响应度
        gamma0, gamma1 = E_step(dataSetArr, alpha0, mu0, sigmod0, alpha1, mu1, sigmod1)
        #步骤3：M步
        mu0, mu1, sigmod0, sigmod1, alpha0, alpha1 = \
            M_step(mu0, mu1, gamma0, gamma1, dataSetArr)

    #迭代结束后将更新后的各参数返回
    return alpha0, mu0, sigmod0, alpha1, mu1, sigmod1

if __name__ == '__main__':
    start = time.time()

    #设置两个高斯模型进行混合，这里是初始化两个模型各自的参数
    #见“9.3 EM算法在高斯混合模型学习中的应用”
    # alpha是“9.3.1 高斯混合模型” 定义9.2中的系数α
    # mu0是均值μ
    # sigmod是方差σ
    #在设置上两个alpha的和必须为1，其他没有什么具体要求，符合高斯定义就可以
    alpha0 = 0.3; mu0 = -2; sigmod0 = 0.5
    alpha1 = 0.7; mu1 = 0.5; sigmod1 = 1

    #初始化数据集
    dataSetList = loadData(mu0, sigmod0, mu1, sigmod1, alpha0, alpha1)

    #打印设置的参数
    print('---------------------------')
    print('the Parameters set is:')
    print('alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f'%(
        alpha0, alpha1, mu0, mu1, sigmod0, sigmod1
    ))

    #开始EM算法，进行参数估计
    alpha0, mu0, sigmod0, alpha1, mu1, sigmod1 = EM_Train(dataSetList)

    #打印参数预测结果
    print('----------------------------')
    print('the Parameters predict is:')
    print('alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f' % (
        alpha0, mu0, sigmod0, alpha1, mu1, sigmod1
    ))

    #打印时间
    print('----------------------------')
    print('time span:', time.time() - start)
