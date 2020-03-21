# TianChi_Protein-Secondary-Structure-Prediction
## 比赛题目
- 目前已知一部分氨基酸序列和与其对应的二级结构，通过已有数据寻找一级结构到二级结构的映射模型，提高通过氨基酸序列进行蛋白质二级结构预测的准确性。
- 比赛链接：[蛋白质结构预测大赛-天池大赛-阿里云天池](https://tianchi.aliyun.com/competition/entrance/231781/information)
## 文件说明
- model4.py：定义并生成模型
- evaluate4.py：加载训练得到的参数并进行二级结构预测
## 方案介绍
### 特征表示
![avatar](https://github.com/yjh126yjh/TianChi_Protein-Secondary-Structure-Prediction/raw/master/pics/Feature_Representation.png)
### 网络结构
![avatar](https://github.com/yjh126yjh/TianChi_Protein-Secondary-Structure-Prediction/raw/master/pics/CNN.png)