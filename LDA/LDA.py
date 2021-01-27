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
        1：'said game england would time first back play last good'
        2：'said year would economy growth also economic bank government could'
        3：'said year games sales company also market last firm 2004'
        4：'film said music best also people year show number digital'
        5：'said would people government labour election party blair could also'
    运行时长：7620.51s    
'''

import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
import time


#定义加载数据的函数
def load_data(file, K):
    '''
    INPUT:
    file - (str) 数据文件的路径
    K - (int) 设定的话题数
    
    OUTPUT:
    org_topics - (list) 原始话题标签列表
    text - (list) 文本列表
    words - (list) 单词列表
    alpha - (list) 话题概率分布，模型超参数
    beta - (list) 单词概率分布，模型超参数
    
    '''
    df = pd.read_csv(file)  #读取文件
    org_topics = df['category'].unique().tolist()  #保存文本原始的话题标签
    M = df.shape[0]  #文本数
    alpha = np.zeros(K)  #alpha是LDA模型的一个超参数，是对话题概率的预估计，这里取文本数据中各话题的比例作为alpha值，实际可以通过模型训练得到
    beta = np.zeros(1000)  #beta是LDA模型的另一个超参数，是词汇表中单词的概率分布，这里取各单词在所有文本中的比例作为beta值，实际也可以通过模型训练得到
    #计算各话题的比例作为alpha值
    for k, topic in enumerate(org_topics):
        alpha[k] = df[df['category'] == topic].shape[0] / M
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
    words_cnt = np.zeros(len(words))  #用来保存单词的出现频次
    #循环计算words列表中各单词出现的词频
    for i in range(len(text)):
        t = text[i]  #取出第i条文本
        for w in t:
            ind = words.index(w)  #取出第i条文本中的第t个单词在单词列表中的索引
            words_cnt[ind] += 1  #对应位置的单词出现频次加一
    sort_inds = np.argsort(words_cnt)[::-1]  #对单词出现频次降序排列后取出其索引值
    words = [words[ind] for ind in sort_inds[:1000]]  #将出现频次前1000的单词保存到words列表
    #去除文本text中不在词汇表words中的单词
    for i in range(len(text)):
        t = []
        for w in text[i]:
            if w in words:
                ind = words.index(w)
                t.append(w)
                beta[ind] += 1  #统计各单词在文本中的出现频次
        text[i] = t
    beta /= np.sum(beta)  #除以文本的总单词数得到各单词所占比例，作为beta值
    return org_topics, text, words, alpha, beta


#定义潜在狄利克雷分配函数，采用收缩的吉布斯抽样算法估计模型的参数theta和phi
def do_lda(text, words, alpha, beta, K, iters):
    '''
    INPUT:
    text - (list) 文本列表
    words - (list) 单词列表
    alpha - (list) 话题概率分布，模型超参数
    beta - (list) 单词概率分布，模型超参数
    K - (int) 设定的话题数
    iters - (int) 设定的迭代次数
    
    OUTPUT:
    theta - (array) 话题的条件概率分布p(zk|dj)，这里写成p(zk|dj)是为了和PLSA模型那一章的符号统一一下，方便对照着看
    phi - (array) 单词的条件概率分布p(wi|zk)
    
    '''
    M = len(text)  #文本数
    V = len(words)  #单词数
    N_MK = np.zeros((M, K))  #文本-话题计数矩阵
    N_KV = np.zeros((K, V))  #话题-单词计数矩阵
    N_M = np.zeros(M)  #文本计数向量
    N_K = np.zeros(K)  #话题计数向量
    Z_MN = []  #用来保存每条文本的每个单词所在位置处抽样得到的话题
    #算法20.2的步骤(2)，对每个文本的所有单词抽样产生话题，并进行计数
    for m in range(M):
        zm = []
        t = text[m]
        for n, w in enumerate(t):
            v = words.index(w)
            z = np.random.randint(K)
            zm.append(z)
            N_MK[m, z] += 1
            N_M[m] += 1
            N_KV[z, v] += 1
            N_K[z] += 1
        Z_MN.append(zm)
    #算法20.2的步骤(3)，多次迭代进行吉布斯抽样
    for i in range(iters):
        print('{}/{}'.format(i+1, iters))
        for m in range(M):
            t = text[m]
            for n, w in enumerate(t):
                v = words.index(w)
                z = Z_MN[m][n]
                N_MK[m, z] -= 1
                N_M[m] -= 1
                N_KV[z][v] -= 1
                N_K[z] -= 1
                p = []  #用来保存对K个话题的条件分布p(zi|z_i,w,alpha,beta)的计算结果
                sums_k = 0  
                for k in range(K):
                    p_zk = (N_KV[k][v] + beta[v]) * (N_MK[m][k] + alpha[k])  #话题zi=k的条件分布p(zi|z_i,w,alpha,beta)的分子部分
                    sums_v = 0
                    sums_k += N_MK[m][k] + alpha[k]  #累计(nmk + alpha_k)在K个话题上的和
                    for t in range(V):
                        sums_v += N_KV[k][t] + beta[t]  #累计(nkv + beta_v)在V个单词上的和
                    p_zk /= sums_v
                    p.append(p_zk)
                p = p / sums_k
                p = p / np.sum(p)  #对条件分布p(zi|z_i,w,alpha,beta)进行归一化，保证概率的总和为1
                new_z = np.random.choice(a=K, p=p)  #根据以上计算得到的概率进行抽样，得到新的话题
                Z_MN[m][n] = new_z  #更新当前位置处的话题为上面抽样得到的新话题
                #更新计数
                N_MK[m, new_z] += 1
                N_M[m] += 1
                N_KV[new_z, v] += 1
                N_K[new_z] += 1
    #算法20.2的步骤(4)，利用得到的样本计数，估计模型的参数theta和phi
    theta = np.zeros((M, K))
    phi = np.zeros((K, V))
    for m in range(M):
        sums_k = 0
        for k in range(K):
            theta[m, k] = N_MK[m][k] + alpha[k]  #参数theta的分子部分
            sums_k += theta[m, k]  #累计(nmk + alpha_k)在K个话题上的和，参数theta的分母部分
        theta[m] /= sums_k  #计算参数theta
    for k in range(K):
        sums_v = 0
        for v in range(V):
            phi[k, v] = N_KV[k][v] + beta[v]  #参数phi的分子部分
            sums_v += phi[k][v]  #累计(nkv + beta_v)在V个单词上的和，参数phi的分母部分
        phi[k] /= sums_v  #计算参数phi
    return theta, phi


if __name__ == "__main__":
    K = 5  #设定话题数为5
    org_topics, text, words, alpha, beta = load_data('bbc_text.csv', K)  #加载数据
    print('Original Topics:')
    print(org_topics)  #打印原始的话题标签列表
    start = time.time()  #保存开始时间
    iters = 10  #为了避免运行时间过长，这里只迭代10次，实际上10次是不够的，要迭代足够的次数保证吉布斯抽样进入燃烧期，这样得到的参数才能尽可能接近样本的实际概率分布
    theta, phi = do_lda(text, words, alpha, beta, K, iters)  #LDA的吉布斯抽样
    #打印出每个话题zk条件下出现概率最大的前10个单词，即P(wi|zk)在话题zk中最大的10个值对应的单词，作为对话题zk的文本描述
    for k in range(K):
        sort_inds = np.argsort(phi[k])[::-1]  #对话题zk条件下的P(wi|zk)的值进行降序排列后取出对应的索引值
        topic = []  #定义一个空列表用于保存话题zk概率最大的前10个单词
        for i in range(10):
            topic.append(words[sort_inds[i]])  
        topic = ' '.join(topic)  #将10个单词以空格分隔，构成对话题zk的文本表述
        print('Topic {}: {}'.format(k+1, topic))  #打印话题zk
    end = time.time()
    print('Time:', end-start)