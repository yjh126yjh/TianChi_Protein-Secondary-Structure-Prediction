# TianChi_Protein-Secondary-Structure-Prediction
## 比赛题目
- 目前已知一部分氨基酸序列和与其对应的二级结构，通过已有数据寻找一级结构到二级结构的映射模型，提高通过氨基酸序列进行蛋白质二级结构预测的准确性。
- 比赛链接：[蛋白质结构预测大赛-天池大赛-阿里云天池](https://tianchi.aliyun.com/competition/entrance/231781/information)
- 天池Notebook：[蛋白质结构预测大赛TOP3团队方案分享-天池实验室-阿里云天池](https://tianchi.aliyun.com/notebook-ai/detail?postId=98092)
## 文件说明
- model.py：定义并生成模型
- evaluate.py：加载训练得到的参数并进行二级结构预测
- preprocess.py：预处理数据
- dataset.py：划分数据集
- whole_sequence-best.hdf5.xz：已经训练好的网络参数，线上成绩0.772
- train.py：训练模型
## 方案介绍
### 特征表示
![](https://github.com/yjh126yjh/TianChi_Protein-Secondary-Structure-Prediction/raw/master/pics/Feature_Representation.png)
### 网络结构
![](https://github.com/yjh126yjh/TianChi_Protein-Secondary-Structure-Prediction/raw/master/pics/CNN.png)
### 结果复现
1. 下载model.py、evaluate.py、dataset.py，下载whole_sequence-best.hdf5.xz并解压，放在同一个文件夹下。
2. 将evaluate.py中的filepath改成测试数据的路径，然后执行python evaluate.py。