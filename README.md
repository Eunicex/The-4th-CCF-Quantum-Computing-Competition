# The 4th CCF Quantum Computing Competition

**本仓库为复旦大学代表队“薛旦谔的猫”参加第四届 CCF 量子计算大赛复赛的参赛代码。最终成绩为 85 分。**

## 题目一

详见 [`Question1`](./Question1) 文件夹中的[文档](./Question1/Q1Document/问题一文档.pdf)说明。

## 题目二

本题要求训练一个机器学习模型，根据九项空气质量相关指标预测空气质量等级（四分类任务）。  
需要在两个文件中分别完成模型设计与训练：

- 在 [`vqnet_model.py`](./Question2/vqnet_model.py) 中实现经典机器学习模型；
- 在 [`quantum_model.py`](./Question2/quantum_model.py) 中实现量子机器学习模型，并与经典模型进行对比分析。

训练集为 [`train_data.csv`](./Question2/train_data.csv)，测试集为 [`test_data.csv`](./Question2/test_data.csv)。

我们的量子模型参考了 [VQNet 官方教程文档中“混合量子经典迁移模型”](https://qcloud.originqc.com.cn/document/vqnet_api_cn/rst/qml_demo.html#id16)。最终在测试集上取得了 **0.9380** 的准确率。