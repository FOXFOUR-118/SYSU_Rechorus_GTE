# SYSU_Rechorus_GTE
Sun Yat-sen University Artificial Intelligence College Machine Learning Course Project
## 简介
这是一个基于Rechorus2.0框架复现的GTE算法，GTE的介绍论文如下：https://arxiv.org/pdf/2308.11127.pdf
## 安装
```bash
# 需要您在命令行中输入以下命令以安装依赖
pip install -r requirements.txt
```
## 使用方法
1、要运行GTE_Rechorus，您首先需要Grocery_and_Gourmet_Food和MIND_Large中的数据集，这是Rechorus2.0框架中提供的两个数据集，请按照对应目录里的代码获取并进行预处理  
2、我们编写的GTE.py放在src/model/general目录下  
3、进入src目录，在命令行中输入如下命令以运行：
```bash
python main.py --model_name GTE --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food
python main.py --model_name GTE --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MIND_Large
```
## 实验
我们小组基于Rechorus2.0框架复现了GTE算法，并使用ReChorus框架里的两个数据集Grocery_and_Gourmet_Food、MIND_Large来与原ReChorus中同类别的两个其他模型：BUIR和NeuMF进行对比。
