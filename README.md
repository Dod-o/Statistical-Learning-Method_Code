## 插一句嘴
我搭了一个类似每日arxiv的网站，将paper汉化（不要钱，纯**为爱发电**），目前标题和摘要都用gpt翻译成中文了，早上扫一眼新出的paper将会更加方便。

近期将会推出论文全文汉化版，一天百篇paper不是梦！

链接： [学术巷子(xueshuxiangzi.com)](https://www.xueshuxiangzi.com/)


前言
====

力求每行代码都有注释，重要部分注明公式来源。具体会追求下方这样的代码，学习者可以照着公式看程序，让代码有据可查。

![image](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/CodePic.png)

    
如果时间充沛的话，可能会试着给每一章写一篇博客。先放个博客链接吧：[传送门](http://www.pkudodo.com/)。    

##### 注：其中Mnist数据集已转换为csv格式，由于体积为107M超过限制，改为压缩包形式。下载后务必先将Mnist文件内压缩包直接解压。  

### 【Updates】
**书籍出版**：目前已与**人民邮电出版社**签订合同，未来将结合该repo整理出版机器学习实践相关书籍。同时会在book分支中对代码进行重构，欢迎在issue中提建议！同时issue中现有的问题也会考虑进去。（Feb 12 2022）

**线下培训**：女朋友计划近期开办**ML/MLP/CV线下培训班**，地点**北上广深杭**，目标各方向**快速入门**，正在筹备。这里帮她打个广告，可以添加微信15324951814（备注线下培训）。本人也会被拉过去义务评估课程质量。。。（Feb 12 2022）

**无监督部分更新**：部分**无监督**算法已更新！！！ 该部分由[Harold-Ran](https://github.com/Harold-Ran)提供，在此感谢！ 有其他算法补充的同学也欢迎添加我微信并pr！（Jan 27 2021）
       
实现
======

## 监督部分

### 第二章 感知机：
博客：[统计学习方法|感知机原理剖析及实现](http://www.pkudodo.com/2018/11/18/1-4/)      
实现：[perceptron/perceptron_dichotomy.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/perceptron/perceptron_dichotomy.py)
      
### 第三章 K近邻：
博客：[统计学习方法|K近邻原理剖析及实现](http://www.pkudodo.com/2018/11/19/1-2/)      
实现：[KNN/KNN.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/KNN/KNN.py)
      
### 第四章 朴素贝叶斯：
博客：[统计学习方法|朴素贝叶斯原理剖析及实现](http://www.pkudodo.com/2018/11/21/1-3/)      
实现：[NaiveBayes/NaiveBayes.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/NaiveBayes/NaiveBayes.py)    
      
### 第五章 决策树：
博客：[统计学习方法|决策树原理剖析及实现](http://www.pkudodo.com/2018/11/30/1-5/)      
实现：[DecisionTree/DecisionTree.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/DecisionTree/DecisionTree.py)    
      
### 第六章 逻辑斯蒂回归与最大熵模型：       
博客：逻辑斯蒂回归：[统计学习方法|逻辑斯蒂原理剖析及实现](http://www.pkudodo.com/2018/12/03/1-6/)        
博客：最大熵：[统计学习方法|最大熵原理剖析及实现](http://www.pkudodo.com/2018/12/05/1-7/)        

实现：逻辑斯蒂回归：[Logistic_and_maximum_entropy_models/logisticRegression.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/Logistic_and_maximum_entropy_models/logisticRegression.py)    
实现：最大熵：[Logistic_and_maximum_entropy_models/maxEntropy.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/Logistic_and_maximum_entropy_models/maxEntropy.py)       
      
### 第七章 支持向量机：    
博客：[统计学习方法|支持向量机(SVM)原理剖析及实现](http://www.pkudodo.com/2018/12/16/1-8/)      
实现：[SVM/SVM.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/SVM/SVM.py)    
      
### 第八章 提升方法：
实现：[AdaBoost/AdaBoost.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/AdaBoost/AdaBoost.py)    
      
### 第九章 EM算法及其推广：
实现：[EM/EM.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/EM/EM.py)    
      
### 第十章 隐马尔可夫模型：
实现：[HMM/HMM.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/HMM/HMM.py)    

## 无监督部分

### 第十四章 聚类方法
实现：[K-means_Clustering.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/Clustering/K-means_Clustering/K-means_Clustering.py)

实现：[Hierachical_Clustering.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/Clustering/Hierachical_Clustering/Hierachical_Clustering.py)

### 第十六章 主成分分析
实现：[PCA.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/PCA/PCA.py)

### 第十七章 潜在语意分析
实现：[LSA.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/LSA/LSA.py)

### 第十八章 概率潜在语意分析
实现：[PLSA.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/PLSA/PLSA.py)

### 第二十章 潜在狄利克雷分配
实现：[LDA.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/LDA/LDA.py)

### 第二十一章 PageRank算法
实现：[Page_Rank.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/Page_Rank/Page_Rank.py)




## 许可 / License
本项目内容许可遵循[Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)。

The content of this project itself is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


联系
======
欢迎pr，有疑问也可通过issue、微信或邮件联系。      
此外如果有需要**MSRA**实习内推的同学，欢迎骚扰。             
**Wechat:** lvtengchao（备注“blog-学校/单位-姓名”）      
**Email:** lvtengchao@pku.edu.cn      
      
