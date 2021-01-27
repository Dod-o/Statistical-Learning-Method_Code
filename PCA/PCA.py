#coding=utf-8
#Author:Harold
#Date:2021-1-27
#Email:zenghr_zero@163.com

'''
数据集：cars
数据集数量：387
-----------------------------
运行结果：
    主成分个数：3
    可解释偏差：0.71
    运行时长：0.14s    
'''

import numpy as np
import pandas as pd
import time 


#定义加载数据的函数
def load_data(file):
    '''
    INPUT:
    file - (str) 数据文件的路径
    
    OUTPUT:
    df - (dataframe) 读取的数据表格
    X - (array) 特征数据数组
    
    '''
    df = pd.read_csv(file)  #读取csv文件
    df.drop('Sports', axis=1, inplace=True)  #去掉类别数据
    X = np.asarray(df.values).T  #将数据转换成数组
    return df, X


#定义规范化函数，对每一列特征进行规范化处理，使其成为期望为0方差为1的标准分布
def Normalize(X):
    '''
    INPUT:
    X - (array) 特征数据数组
    
    OUTPUT:
    X - (array) 规范化处理后的特征数据数组
    
    '''
    m, n = X.shape
    for i in range(m):
        E_xi = np.mean(X[i])  #第i列特征的期望
        Var_xi = np.var(X[i], ddof=1)  #第i列特征的方差
        for j in range(n):
            X[i][j] = (X[i][j] - E_xi) / np.sqrt(Var_xi)  #对第i列特征的第j条数据进行规范化处理
    return X


#定义奇异值分解函数，计算V矩阵和特征值
def cal_V(X):
    '''
    INPUT:
    X - (array) 特征数据数组
    
    OUTPUT:
    eigvalues - (list) 特征值列表，其中特征值按从大到小排列
    V - (array) V矩阵
    
    '''
    newX = X.T / np.sqrt(X.shape[1]-1)  #构造新矩阵X'
    Sx = np.matmul(newX.T, newX)  #计算X的协方差矩阵Sx = X'.T * X'
    V_T = []  #用于保存V的转置
    w, v = np.linalg.eig(Sx)  #计算Sx的特征值和对应的特征向量，即为X’的奇异值和奇异向量
    tmp = {}  #定义一个字典用于保存特征值和特征向量，字典的键为特征值，值为对应的特征向量
    for i in range(len(w)):
        tmp[w[i]] = v[i]
    eigvalues = sorted(tmp, reverse=True)  #将特征值逆序排列后保存到eigvalues列表中
    for i in eigvalues:
        d = 0
        for j in range(len(tmp[i])):
            d += tmp[i][j] ** 2
        V_T.append(tmp[i] / np.sqrt(d))  #计算特征值i的单位特征向量，即为V矩阵的列向量，将其保存到V_T中
    V = np.array(V_T).T  #对V_T进行转置得到V矩阵
    return eigvalues, V


#定义主成分分析函数
def do_pca(X, k):
    '''
    INPUT:
    X - (array) 特征数据数组
    k - (int) 设定的主成分个数
    
    OUTPUT:
    fac_load - (array) 因子负荷量数组
    dimrates - (list) 可解释偏差列表
    Y - (array) 主成分矩阵
    
    '''
    eigvalues, V = cal_V(X)  #计算特征值和V矩阵
    Vk = V[:, :k]  #取V矩阵的前k列
    Y = np.matmul(Vk.T, X)  #计算主成分矩阵，将m*n的样本矩阵X转换成k*n的样本主成分矩阵
    dimrates = [i / sum(eigvalues) for i in eigvalues[:k]]  #计算可解释偏差，即前k个奇异值中每个奇异值占奇异值总和的比例，这个比例表示主成分i可解释原始数据中的可变性的比例
    fac_load = np.zeros((k, X.shape[0]))  #用来保存主成分的因子负荷量
    for i in range(k): 
        for j in range(X.shape[0]):
            fac_load[i][j] = np.sqrt(eigvalues[i]) * Vk[j][i] / np.sqrt(np.var(X[j]))  #计算主成分i对应原始特征j的因子负荷量，保存到fac_load中
    return fac_load, dimrates, Y


if __name__ == "__main__":
    df, X = load_data('cars.csv')  #加载数据
    start = time.time()  #保存开始时间
    X = Normalize(X)  #对样本数据进行规范化处理
    k = 3  #设定主成分个数为3
    fac_load, dimrates, Y = do_pca(X, k)  #进行主成分分析
    pca_result = pd.DataFrame(fac_load, index=['Dimension1', 'Dimension2', 'Dimension3'], columns=df.columns)  #将结果保存为dataframe格式
    pca_result['Explained Variance'] = dimrates  #将可解释偏差保存到pca_result的'Explained Variance'列
    end = time.time()  #保存结束时间
    print('Time:', end-start)