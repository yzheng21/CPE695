from math import log
import operator
def createDataset():
    dataset = [[0, 0, 0, 0, 'N'],
               [0, 0, 0, 1, 'N'],
               [1, 0, 0, 0, 'Y'],
               [2, 1, 0, 0, 'Y'],
               [2, 2, 1, 0, 'Y'],
               [2, 2, 1, 1, 'N'],
               [1, 2, 1, 1, 'Y']]
    labels = ['outlook','temperature','humidity','windy']
    return dataset, labels

#计算熵
def calentropy(dataset):
    numentries = len(dataset)
    labelcounts = {}
    for feature in dataset:
        currentlabel = feature[-1]
        if currentlabel not in labelcounts.keys():
            labelcounts[currentlabel] = 0
        labelcounts[currentlabel] +=1

    entropy = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key])/numentries
        entropy -= prob*log(prob,2)
    return entropy

#选择最大的gain ratio所在的feature
def choosefeature(dataset):
    numfeatures = len(dataset[0])-1
    wholeentropy = calentropy(dataset)
    bestinfogainratio = 0.0
    bestfeature = -1
    for i in range(numfeatures):
        featurelist = [example[i] for example in dataset]  #[0, 0, 1, 2, 2, 2, 1]
        uniqueval = set(featurelist)   #{0, 1, 2}
        newentropy = 0.0
        splitinfo = 0.0
        for value in uniqueval:
            subdataset = splitdataset(dataset,i,value)    #每个唯一值对应的子集
            prob = len(subdataset)/float(len(dataset))
            newentropy += prob*calentropy(subdataset)
            splitinfo += -prob*log(prob,2)    #每个特征的信息熵
        infogain = wholeentropy - newentropy
        if (splitinfo == 0):
            continue
        infogainratio = infogain/splitinfo

        if (infogainratio>bestinfogainratio):
            bestinfogainratio = infogainratio
            bestfeature = i
    return bestfeature

def splitdataset(dataset,axis,value):
    retdataset = []
    for feature in dataset:
        if feature[axis] == value:
            reducefeature = feature[:axis]
            reducefeature.extend(feature[axis+1:])
            retdataset.append(reducefeature)
    return retdataset

def majoritycnt(classlist):
    classcount = {}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote] = 0
        classcount[vote] += 1
    sortedClassCount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) ==len(classList):#类别相同则停止划分
        return classList[0]
    if len(dataSet[0]) == 1:#所有特征已经用完
        return majoritycnt(classList)
    bestFeat = choosefeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]#为了不改变原始列表的内容复制了一下
        myTree[bestFeatLabel][value] = createTree(splitdataset(dataSet,
                                        bestFeat, value),subLabels)
    return myTree

if __name__=='__main__':
    dataset,labels=createDataset()
    featlabels=[]
    mytree = createTree(dataset,labels)
    print(mytree)

