# 基于Bert的关系抽取和实体识别
BERT is used for Relation Extraction and Entity Extract 

2019 CCF大数据与计算智能大赛（[训练赛-文本实体识别及关系抽取](https://www.datafountain.cn/competitions/371)），利用经过特定处理的公共数据集SemEval2010，数据集中的文本共包括9种实体关系
**任务介绍**
* 对句子进行实体抽取
* 并根据语义及其他信息来判断实体之间的关系

**模型介绍**：
* 本目录[Detail.pdf](https://github.com/Beleiaya/EntityExtract/blob/master/Detail.pdf)

**比赛结果**：
* 级联模型：accuracy=0.33,排名10/1572(12月30日排名)
* 联合模型：accuracy=0.28,排名11/1572(12月30日排名)



## 文件目录

|name|function|
|-|-|
|content/data4relation_2 |数据存储文件|
|run_classifier_predicate.py |基于Bert的关系分类|
|NER4Relation.py| 基于Bert的实体抽取（序列标注方法）|
|run_classifier_joint.py| 基于Bert的联合训练模型|
|Detail.pdf |模型详细介绍 |


## 环境要求
+ python 3.6+
+ Tensorflow 1.12.0+
+ Bert-base(uncased)

## 模型训练
### 基于Bert的级联模型
#### （1）基于Bert关系分类模型训练
```
python run_classifier_predicate.py
```
#### （2）基于Bert实体抽取（序列标注方法）模型训练
```
python NER4Relation.py 
```
### 基于Bert的联合训练模型
```
python run_classifier_joint.py
```
