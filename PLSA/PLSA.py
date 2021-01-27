#coding=utf-8
#Author:Harold
#Date:2021-1-27
#Email:zenghr_zero@163.com

'''
数据集：bbc_text
数据集数量：2225
-----------------------------
运行结果：
    话题数：5
    原始话题：'tech', 'business', 'sport', 'entertainment', 'politics'
    生成话题：
        1：'said year government people mobile last number growth phone market'
        2：'said people film could would also technology made make government'
        3：'said would could best music also world election labour people'
        4：'said first england also time game players wales would team'
        5：'said also would company year world sales firm market last'
    运行时长：531.13s    
'''

import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
import time


#定义加载数据的函数
def load_data(file):
    '''
    INPUT:
    file - (str) 数据文件的路径
    
    OUTPUT:
    org_topics - (list) 原始话题标签列表
    text - (list) 文本列表
    words - (list) 单词列表
    
    '''
    df = pd.read_csv(file)  #读取文件
    org_topics = df['category'].unique().tolist()  #保存文本原始的话题标签
    df.drop('category', axis=1, inplace=True)
    n = df.shape[0]  #n为文本数量
    text = []
    words = []
    for i in df['text'].values:
        t = i.translate(str.maketrans('', '', string.punctuation))  #去除文本中的标点符号
        t = [j for j in t.split() if j not in stopwords.words('english')]  #去除文本中的停止词
        t = [j for j in t if len(j) > 3]  #长度小于等于3的单词大多是无意义的，直接去除
        text.append(t)  #将处理后的文本保存到文本列表中
        words.extend(set(t))  #将文本中所包含的单词保存到单词列表中
    words = list(set(words))  #去除单词列表中的重复单词
    return org_topics, text, words


#定义构建单词-文本矩阵的函数，这里矩阵的每一项表示单词在文本中的出现频次，也可以用TF-IDF来表示
def frequency_counter(text, words):
    '''
    INPUT:
    text - (list) 文本列表
    words - (list) 单词列表
    
    OUTPUT:
    words - (list) 出现频次为前1000的单词列表
    X - (array) 单词-文本矩阵
    
    '''
    words_cnt = np.zeros(len(words))  #用来保存单词的出现频次
    X = np.zeros((1000, len(text)))  #定义m*n的矩阵，其中m为单词列表中的单词个数，为避免运行时间过长，这里只取了出现频次为前1000的单词，因此m为1000，n为文本个数
    #循环计算words列表中各单词出现的词频
    for i in range(len(text)):
        t = text[i]  #取出第i条文本
        for w in t:
            ind = words.index(w)  #取出第i条文本中的第t个单词在单词列表中的索引
            words_cnt[ind] += 1  #对应位置的单词出现频次加一
    sort_inds = np.argsort(words_cnt)[::-1]  #对单词出现频次降序排列后取出其索引值
    words = [words[ind] for ind in sort_inds[:1000]]  #将出现频次前1000的单词保存到words列表
    #构建单词-文本矩阵
    for i in range(len(text)):
        t = text[i]  #取出第i条文本
        for w in t:
            if w in words:  #如果文本t中的单词w在单词列表中，则将X矩阵中对应位置加一
                ind = words.index(w)
                X[ind, i] += 1
    return words, X


#定义概率潜在语义分析函数，采用EM算法进行PLSA模型的参数估计
def do_plsa(X, K, words, iters = 10):
    '''
    INPUT:
    X - (array) 单词-文本矩阵
    K - (int) 设定的话题数
    words - (list) 出现频次为前1000的单词列表
    iters - (int) 设定的迭代次数
    
    OUTPUT:
    P_wi_zk - (array) 话题zk条件下产生单词wi的概率数组
    P_zk_dj - (array) 文本dj条件下属于话题zk的概率数组
    
    '''
    M, N = X.shape  #M为单词数，N为文本数
    #P_wi_zk表示P(wi|zk)，是一个K*M的数组，其中每个值表示第k个话题zk条件下产生第i个单词wi的概率，这里将每个值随机初始化为0-1之间的浮点数
    P_wi_zk = np.random.rand(K, M)
    #对于每个话题zk，保证产生单词wi的概率的总和为1
    for k in range(K):
        P_wi_zk[k] /= np.sum(P_wi_zk[k])
    #P_zk_dj表示P(zk|dj)，是一个N*K的数组，其中每个值表示第j个文本dj条件下产生第k个话题zk的概率，这里将每个值随机初始化为0-1之间的浮点数
    P_zk_dj = np.random.rand(N, K)
    #对于每个文本dj，属于话题zk的概率的总和为1
    for n in range(N):
        P_zk_dj[n] /= np.sum(P_zk_dj[n])
    #P_zk_wi_dj表示P(zk|wi,dj)，是一个M*N*K的数组，其中每个值表示在单词-文本对(wi,dj)的条件下属于第k个话题zk的概率，这里设置初始值为0
    P_zk_wi_dj = np.zeros((M, N, K))
    #迭代执行E步和M步
    for i in range(iters):
        print('{}/{}'.format(i+1, iters))  
        #执行E步
        for m in range(M):
            for n in range(N):
                sums = 0
                for k in range(K):
                    P_zk_wi_dj[m, n, k] = P_wi_zk[k, m] * P_zk_dj[n, k]  #计算P(zk|wi,dj)的分子部分，即P(wi|zk)*P(zk|dj)
                    sums += P_zk_wi_dj[m, n, k]  #计算P(zk|wi,dj)的分母部分，即P(wi|zk)*P(zk|dj)在K个话题上的总和
                P_zk_wi_dj[m, n, :] = P_zk_wi_dj[m, n, :] / sums  #得到单词-文本对(wi,dj)条件下的P(zk|wi,dj)
        #执行M步，计算P(wi|zk)
        for k in range(K):
            s1 = 0
            for m in range(M):
                P_wi_zk[k, m] = 0
                for n in range(N):
                    P_wi_zk[k, m] += X[m, n] * P_zk_wi_dj[m, n, k]  #计算P(wi|zk)的分子部分，即n(wi,dj)*P(zk|wi,dj)在N个文本上的总和，其中n(wi,dj)为单词-文本矩阵X在文本对(wi,dj)处的频次
                s1 += P_wi_zk[k, m]  #计算P(wi|zk)的分母部分，即n(wi,dj)*P(zk|wi,dj)在N个文本和M个单词上的总和
            P_wi_zk[k, :] = P_wi_zk[k, :] / s1  #得到话题zk条件下的P(wi|zk)
        #执行M步，计算P(zk|dj)
        for n in range(N):
            for k in range(K):
                P_zk_dj[n, k] = 0
                for m in range(M):
                    P_zk_dj[n, k] += X[m, n] * P_zk_wi_dj[m, n, k]  #同理计算P(zk|dj)的分子部分，即n(wi,dj)*P(zk|wi,dj)在N个文本上的总和
                P_zk_dj[n, k] = P_zk_dj[n, k] / np.sum(X[:, n])  #得到文本dj条件下的P(zk|dj)，其中n(dj)为文本dj中的单词个数，由于我们只取了出现频次前1000的单词，所以这里n(dj)计算的是文本dj中在单词列表中的单词数
    return P_wi_zk, P_zk_dj


if __name__ == "__main__":
    org_topics, text, words = load_data('bbc_text.csv')  #加载数据
    print('Original Topics:')
    print(org_topics)  #打印原始的话题标签列表
    start = time.time()  #保存开始时间
    words, X = frequency_counter(text, words)  #取频次前1000的单词重新构建单词列表，并构建单词-文本矩阵
    K = 5  #设定话题数为5
    P_wi_zk, P_zk_dj = do_plsa(X, K, words, iters = 10)  #采用EM算法对PLSA模型进行参数估计
    #打印出每个话题zk条件下出现概率最大的前10个单词，即P(wi|zk)在话题zk中最大的10个值对应的单词，作为对话题zk的文本描述
    for k in range(K):
        sort_inds = np.argsort(P_wi_zk[k])[::-1]  #对话题zk条件下的P(wi|zk)的值进行降序排列后取出对应的索引值
        topic = []  #定义一个空列表用于保存话题zk概率最大的前10个单词
        for i in range(10):
            topic.append(words[sort_inds[i]])  
        topic = ' '.join(topic)  #将10个单词以空格分隔，构成对话题zk的文本表述
        print('Topic {}: {}'.format(k+1, topic))  #打印话题zk
    end = time.time()
    print('Time:', end-start)