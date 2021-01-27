#coding=utf-8
#Author:Harold
#Date:2021-1-27
#Email:zenghr_zero@163.com

'''
有向图：directed_graph.png
结点数量：7
-----------------------------
运行结果：
    迭代算法：
        迭代次数：24
        PageRank:   [[0.17030305]
                    [0.10568394]
                    [0.11441021]
                    [0.10629792]
                    [0.10568394]
                    [0.15059975]
                    [0.24702119]]
        运行时长：0.0010s    
    幂法：
        迭代次数：25
        PageRank:   [[0.18860772]
                    [0.09038084]
                    [0.0875305 ]
                    [0.07523049]
                    [0.09038084]
                    [0.15604764]
                    [0.31182196]]
        运行时长：0.0020s  
'''

import numpy as np
import time


#PageRank的迭代算法
def iter_method(n, d, M, R0, eps):
    t = 0  #用来累计迭代次数
    R = R0  #对R向量进行初始化
    judge = False  #用来判断是否继续迭代
    while not judge:
        next_R = d * np.matmul(M, R) + (1 - d) / n * np.ones((7, 1))  #计算新的R向量
        diff = np.linalg.norm(R - next_R)  #计算新的R向量与之前的R向量之间的距离，这里采用的是欧氏距离
        if diff < eps:  #若两向量之间的距离足够小
            judge = True  #则停止迭代
        R = next_R  #更新R向量
        t += 1  #迭代次数加一
    R = R / np.sum(R)  #对R向量进行规范化，保证其总和为1，表示各节点的概率分布
    return t, R


def power_method(n, d, M, R0, eps):
    t = 0  #用来累计迭代次数
    x = R0  #对x向量进行初始化
    judge = False  #用来判断是否继续迭代
    A = d * M + (1 - d) / n * np.eye(n)  #计算A矩阵，其中np.eye(n)用来创建n阶单位阵E
    while not judge:
        next_y = np.matmul(A, x)  #计算新的y向量
        next_x = next_y / np.linalg.norm(next_y)  #对新的y向量规范化得到新的x向量
        diff = np.linalg.norm(x - next_x)  #计算新的x向量与之前的x向量之间的距离，这里采用的是欧氏距离
        if diff < eps:  #若两向量之间的距离足够小
            judge = True  #则停止迭代
            R = x  #得到R向量
        x = next_x  #更新x向量
        t += 1  #迭代次数加一
    R = R / np.sum(R)  #对R向量进行规范化，保证其总和为1，表示各节点的概率分布
    return t, R


if __name__ == "__main__":
    n = 7  #有向图中一共有7个节点
    d = 0.85  #阻尼因子根据经验值确定，这里我们随意给一个值
    M = np.array([[0, 1/4, 1/3, 0, 0, 1/2, 0],
                [1/4, 0, 0, 1/5, 0, 0, 0],
                [0, 1/4, 0, 1/5, 1/4, 0, 0],
                [0, 0, 1/3, 0, 1/4, 0, 0],
                [1/4, 0, 0, 1/5, 0, 0, 0],
                [1/4, 1/4, 0, 1/5, 1/4, 0, 0],
                [1/4, 1/4, 1/3, 1/5, 1/4, 1/2, 0]])  #根据有向图中各节点的连接情况写出转移矩阵
    R0 = np.full((7, 1), 1/7)  #设置初始向量R0，R0是一个7*1的列向量，因为有7个节点，我们把R0的每一个值都设为1/7
    eps = 0.000001  #设置计算精度

    start = time.time()  #保存开始时间
    t, R = iter_method(n, d, M, R0, eps)
    end = time.time()  #保存结束时间
    print('-------PageRank的迭代算法-------')
    print('迭代次数：', t)
    print('PageRank: \n', R)
    print('Time:', end-start)

    start = time.time()  #保存开始时间
    t, R = power_method(n, d, M, R0, eps)
    end = time.time()  #保存结束时间
    print('-------PageRank的幂法-------')
    print('迭代次数：', t)
    print('PageRank: \n', R)
    print('Time:', end-start)