# coding=utf-8
'''
1：决策树原理
       决策树是通过一系列规则对数据进行分类的过程，他提供一种在什么条件下会得到什么值的类似规则方法，
       决策树分为分类树和回归树，分类树对离散变量做决策树，回归树对连续变量做决策树
如果不考虑效率等，那么样本所有特征的判断级联起来终会将某一个样本分到一个类终止块上。
实际上，样本所有特征中有一些特征在分类时起到决定性作用，决策树的构造过程就是找到这些具有决定性作用的特征，
根据其决定性程度来构造一个倒立的树–决定性作用最大的那个特征作为根节点，然后递归找到各分支下子数据集中次大的决定性特征，
直至子数据集中所有数据都属于同一类。所以，构造决策树的过程本质上就是根据数据特征将数据集分类的递归过程，
我们需要解决的第一个问题就是，当前数据集上哪个特征在划分数据分类时起决定性作用。

为了找到决定性的特征、划分出最好的结果，我们必须评估数据集中蕴含的每个特征，
寻找分类数据集的最好特征。完成评估之后，原始数据集就被划分为几个数据子集。
这些数据子集会分布在第一个决策点的所有分支上。如果某个分支下的数据属于同一类型，
则则该分支处理完成，称为一个叶子节点，即确定了分类。如果数据子集内的数据不属于同一类型，
则需要重复划分数据子集的过程。如何划分数据子集的算法和划分原始数据集的方法相同，
直到所有具有相同类型的数据均在一个数据子集内（叶子节点）。

2：决策树的构造过程
         一般包含三个部分
         1、特征选择：特征选择是指从训练数据中众多的特征中选择一个特征作为当前节点的分裂标准，
         如何选择特征有着很多不同量化评估标准标准，从而衍生出不同的决策树算法。
          2、决策树生成： 根据选择的特征评估标准，从上至下递归地生成子节点，
          直到数据集不可分则停止决策树停止生长。 树结构来说，递归结构是最容易理解的方式。
         3、剪枝：决策树容易过拟合，一般来需要剪枝，缩小树结构规模、缓解过拟合。剪枝技术有预剪枝和后剪枝两种。
        核心伪代码如下：
           检测数据集中的每个子项是否属于同一类：
           If so return 类标签
           else
                寻找划分数据集的最好特征
                划分数据集
                创建分支节点
                    for 每个划分的子集
                         调用createBranch函数并增加返回结果到分支节点中
              return 分支节点
3：决策树的优缺点
     决策树适用于数值型和标称型（离散型数据，变量的结果只在有限目标集中取值），能够读取数据集合，提取一系列数据中蕴含的规则。
     在分类问题中使用决策树模型有很多的优点，决策树计算复杂度不高、便于使用、而且高效，决策树可处理具有不相关特征的数据、
     可很容易地构造出易于理解的规则，而规则通常易于解释和理解。决策树模型也有一些缺点，
     比如处理缺失数据时的困难、过度拟合以及忽略数据集中属性之间的相关性等
'''
from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']  # 分类的属性
    return dataSet, labels


# 计算给定数据的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 获得标签
        # 构造存放标签的字典
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 对应的标签数目+1
    # 计算香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 划分数据集,三个参数为带划分的数据集，划分数据集的特征，特征的返回值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 将相同数据集特征的抽取出来
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet  # 返回一个列表


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    beatFeature = -1
    for i in range(numFeature):
        featureList = [example[i] for example in dataSet]  # 获取第i个特征所有的可能取值
        uniqueVals = set(featureList)  # 从列表中创建集合，得到不重复的所有可能取值ֵ
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  # 以i为数据集特征，value为返回值，划分数据集
            prob = len(subDataSet) / float(len(dataSet))  # 数据集特征为i的所占的比例
            newEntropy += prob * calcShannonEnt(subDataSet)  # 计算每种数据集的信息熵
        infoGain = baseEntropy - newEntropy
        # 计算最好的信息增益，增益越大说明所占决策权越大
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 递归构建决策树
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 排序，True升序
    return sortedClassCount[0][0]  # 返回出现次数最多的


# 创建树的函数代码
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 类别完全相同则停止划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有特征值时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最好的数据集划分方式
    bestFeatLabel = labels[bestFeat]  # 得到对应的标签值
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])  # 清空labels[bestFeat],在下一次使用时清零
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # 递归调用创建决策树函数
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    print createTree(dataSet, labels)