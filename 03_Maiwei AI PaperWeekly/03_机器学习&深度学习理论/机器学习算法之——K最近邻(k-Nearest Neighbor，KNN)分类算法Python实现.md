# 机器学习算法之——K最近邻(k-Nearest Neighbor，KNN)分类算法Python实现

@[TOC](k最近邻算法(k-Nearest Neighbor，KNN ）Python实现)

# 1. 学习环境
windows10 、Anaconda(向初学者推荐这个工具) 中的IDE工具Spyder 、python 3.7。

# 2. K-近邻算法概述
## 2.1. K-近邻算法工作原理
在训练集中数据和标签已知的情况下，输入测试数据，将测试数据的特征与训练集中对应的特征进行相互比较，找到训    练集中与之最为相似的前K个数据，则该测试数据对应的类别就是K个数据中出现次数最多的那个分类。简单来说，k-近邻算法采  用测量不同特征值之间的距离方法进行分类。
     

## 2.2 K-近邻算法优缺点

优点：精度高、对异常值不敏感、无数据输入假定。

缺点：计算复杂度高、空间复杂度高。    

适用数据范围：数值型和标称型。

             

## 2.3.K-近邻算法的一般流程

 (1)  收集数据：可以使用任何合适的方法。

 (2)  准备数据：距离计算所需要的数值，最好是结构化的数据格式。

 (3)  分析数据：可以使用任何可行的方法。

 (4)  训练算法：此步骤不适用于K-近邻算法。

 (5)  测试算法：计算错误率。

 (6)  使用算法：首先需要输入样本数据和结构化的输出结果，然后运行K-近邻算法判定输入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理。

 

# 3. 实施K-近邻算法
## 3.1 k-近邻算法伪代码：            

对未知类别属性的数据集中的每个点依次执行以下操作：

 (1)  计算已知类别数据集中的点与当前点之间的距离；

 (2)  按照距离递增次序排序；

 (3)  选取与当前点距离最小的K个点；

(4)  确定前k个点所在类别的出现频率；

 (5)  返回前k个点出现频率最高的类别作为当前点的预测分类；

 ## 3.2 程序清单：k-近邻算法

使用欧式距离公式计算两个向量点A和B之间的距离： 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200229220725632.png)

```python
def classify0(inX,dataSet,labels,k):
    #获取训练数据集的行数
    dataSetSize=dataSet.shape[0]
    #---------------欧氏距离计算-----------------
    #各个函数均是以矩阵形式保存
    #tile():inX沿各个维度的复制次数
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    #.sum()运行加函数，参数axis=1表示矩阵每一行的各个值相加和
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    #--------------------------------------------
    #获取排序（有小到大）后的距离值的索引（序号）
    sortedDistIndicies=distances.argsort()
    #字典，键值对，结构类似于hash表
    classCount={}
    for i in range(k):
        #获取该索引对应的训练样本的标签
        voteIlabel=labels[sortedDistIndicies[i]]
        #累加几类标签出现的次数，构成键值对key/values并存于classCount中
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #Python 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值)元组数组
    #将元组中按照第二列，也就是次数标签，降序排序（由大到小排序）
    sortedClassCount=sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    #返回第一个元素（最高频率）标签
    return sortedClassCount[0][0]
```
# 4. k-近邻算法示例1
示例1：使用K-近邻算法改进约会网站的配对效果

## 4.1 基本流程：

(1)  收集数据：提供文本文件。

(2)  准备数据：使用python解析文本文件。

 (3)  分析数据：使用Matplotlib画二维扩散图。

(4)  训练算法：此步骤不适用于K-近邻算法。

(5)  测试算法：使用海伦提供的部分数据作为测试样本。

测试样本和非测试样本的区别在于：测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。

(6)  使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。

## 4.2 具体实现
### 4.2.1  准备数据：从文本文件中解析数据
将待处理的数据改变为分类器可以接受的格式。该函数的输入为文件名字符串，输出为训练样本矩阵和标签向量

```python
def file2matrix(filename):
    fr = open(filename)
    #readlines()函数一次读取整个文件，readlines() 自动将文件内容分析成一个行的列表，
    #该列表可以由 Python 的 for ... in ... 结构进行处理。
    arrayOLines = fr.readlines()
    #len() 返回字符串、列表、字典、元组等长度。
    #得到文件的行数
    numberOfLines = len(arrayOLines)
    #zeros函数 例:zeros((3,4)),创建3行4列以0填充的矩阵
    #创建以0填充的Numpy 矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        #strip()函数 s.strip(rm) 删除s字符串中开头、结尾处，位于 rm删除序列的字符
        #当rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')
        line = line.strip()  # 截取所有的回车符
        #split()：拆分字符串。通过指定分隔符对字符串进行切片，
        #并返回分割后的字符串列表（list）
        listFromLine = line.split('\t')  #解析文件数据到列表
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
```
测试解析函数文件file2matrix( )

```python
#测试
datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
print(" datingDataMat \n " , datingDataMat," \n")
print(" datingLabels \n" , datingLabels[0:20])
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200229221002160.png#pic_center)

 ### 4.2.2  分析数据：使用Matplotlib创建散点图

在 .py文件开头导入包

```python
import matplotlib
import matplotlib.pyplot as plt
```

```python
fig = plt.figure()  
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()       
```
 散点图使用datingDataMat矩阵的第二、三列数据，分别表示特征值“玩视频游戏所耗时间比”
 
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020022922105288.png#pic_center)

### 4.2.3  准备数据：归一化数值

方程中数字差值最大的属性对计算结果的影响最大，但这三种特征是同等重要的，因此作为三个等权重的特征之一，飞行常客里程数不应该如此严重地影响到计算结果。处理这种不同取值范围的特征时，我们采用的方法是将数值归一化，如将取值范围处理为 0 到 1 或者 -1 到 1 之间。下面公式可以将任意取值范围的特征值转化为 0 到 1 区间内的值：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020022922113937.png)
其中min 和 max 分别是数据集中的最小特征值和最大特征值。

归一化特征值函数代码：

```python
#该函数可以将数字特征值转化为0到1的区间
def autoNorm(dataSet):
    #a.min()返回的就是a中所有元素的最小值
    #a.min(0)返回的就是a的每列最小值
    #a.min(1)返回的是a的每行最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    #生成一个和数据集相同大小的矩阵
    normDataSet = zeros(shape(dataSet))
    #获取数据集行数
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    #特征值相除
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals
 
#测试函数
normMat, ranges, minVals = autoNorm(datingDataMat)
print("normMat: \n", normMat,"\n ")
print("ranges: \n", ranges," \n ")
print("minVals: \n", minVals)
```

测试结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200229221204200.png#pic_center)

### 4.2.4  测试算法：作为完整程序验证分类器
测试代码:

```python
#测试分类器的效果函数
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #获取数据集的行数
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # "\" 换行符
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                       datingLabels[numTestVecs:m], 20)
        print("the classsifier came back with: %d, the real answer is: %d"\
                       %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]): errorCount +=1.0
    print("the total error rate is:%f" % (errorCount/float(numTestVecs)))
 
datingClassTest()   #测试分类器的正确率 测试算法        
```
测试结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200229221246751.png#pic_center)

分类器处理约会数据集的错误率是<font color =red > 6%。

### 4.2.5  使用算法：构建完整可用系统
约会网站预测函数代码:

```python
#使用算法
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    # \ 为换行符
    percentTats = float(input(\
                  "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input ("liters of ice creamm consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-\
                       minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ",\
            resultList[classifierResult - 1])
classifyPerson()    
```
测试结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200229221347985.png#pic_center)

# 5. k-近邻算法示例2
**示例2：手写字识别系统**

## 5.1 基本流程
 (1)  收集数据：提供文本文件。

(2)  准备数据：编写函数img2vector(),将图像格式转化为分类器使用的向量格式。

 (3)  分析数据：在python命令行中检查数据，确保它符合要求。

  (4)  训练算法：此步骤不适用于K-近邻算法。

 (5)  测试算法：编写函数使用提供的部分数据集作为测试样本。 测试样本和非测试样本的区别在于：测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。

  (6)  使用算法：本列没有此步骤，若你感兴趣你可以用此算法去完成 kaggle 上的 Digital Recognition（数字识别）题目。

 ## 5.2 具体实现

### 5.2.1  准备数据：将图像转化为测试向量
转化函数代码:

```python
"""
手写数据集 准备数据：将图像转换为测试向量
"""
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    #返回数组
    return returnVect
#测试函数
testVector = img2vector('testDigits/0_13.txt')
print(testVector[0,0:22])
```

测试结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200229221557145.png#pic_center)    

     

### 5.2.2  测试算法：使用k-近邻算法识别手写数字

测试函数：    
      ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200229221603588.png#pic_center)

错误率为<font color = red > 1.2%。</font>

<br>
<p align=center>
     <strong>- END -</strong>
</p>

<br>

**推荐文章**

[1] [机器学习算法之——走近卷积神经网络(CNN)](https://blog.csdn.net/Charmve/article/details/104872365)<br>
[2] [机器学习算法之——卷积神经网络(CNN)原理讲解](https://blog.csdn.net/Charmve/article/details/104872435)<br>
[3] [卷积神经网络中十大拍案叫绝的操作](https://blog.csdn.net/Charmve/article/details/105093727)<br>
[4] [机器学习算法之——梯度提升(Gradient Boosting)  算法讲解及Python实现](https://blog.csdn.net/Charmve/article/details/103846873)<br>
[5] [机器学习算法之——逻辑回归（Logistic Regression）](https://blog.csdn.net/Charmve/article/details/103844249)<br>
[6] [机器学习算法之——决策树模型(Decision Tree Model)算法讲解及Python实现](https://mp.weixin.qq.com/s?__biz=MzIxMjg1Njc3Mw==&mid=2247484003&idx=6&sn=8be6122009f862a41cc4a313c090bb0f&chksm=97bef8c9a0c971dfc74b0f24a510cffee66d368689d8e97faa54e960c4b6201983b97e488fb3&scene=21&token=2142822614&lang=zh_CN#wechat_redirect)<br>
[7] [机器学习算法之——K最近邻(k-Nearest Neighbor，KNN)分类算法原理讲解](https://blog.csdn.net/Charmve/article/details/103950561)<br>
[8] [机器学习算法之——K最近邻(k-Nearest Neighbor，KNN)算法Python实现](https://blog.csdn.net/Charmve/article/details/104583287)<br>



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200229223432467.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NoYXJtdmU=,size_16,color_FFFFFF,t_70)

<font color = red>**关注微信公众号：迈微AI研习社，回复 “KNN” 获取本博客相关工程及数据文件[Github开源项目]。**

<div align=center>
     <img src="https://img-blog.csdnimg.cn/20200224183312714.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NoYXJtdmU=,size_16,color_FFFFFF,t_70">
</div>
<p align=center>
     △微信扫一扫关注「迈微AI研习社」公众号
</p>

<font color = redd>知识星球：社群旨在分享AI算法岗的秋招/春招准备攻略（含刷题）、面经和内推机会、学习路线、知识题库等。
<div align=center>
  <img src="https://img-blog.csdnimg.cn/20200225182015550.png">
</div>
<p align=center>
     △扫码加入「迈微AI研习社」学习辅导群
</p>

<div align=center>
  <img src="https://img-blog.csdnimg.cn/2020022418361459.png">
</div>

