# Customer Churn Prediction using a Novel Meta-Classifier

## 项目简介

本项目复现了论文[Customer churn prediction using a novel meta-classifier:an investigation on transaction, Telecommunication and customer churn datasets](https://doi.org/10.1007/s10878-024-01196-w)。该项目复现了相关论文的研究成果，通过集成多个基分类器并使用Oracle元分类器来提高客户流失预测的准确性。

## 主要特性

- **多种机器学习算法**: 支持决策树、随机森林、XGBoost、AdaBoost、ExtraTrees等算法
- **两种超参数优化策略**: 
  - SKB + GridSearchCV (SelectKBest + 网格搜索)
  - PFS + BayesSearchCV (Permutation Feature Importance + 贝叶斯搜索)
- **三种模型类型**:
  - 单一模型 (Single Models)
  - 堆叠集成模型 (Stacking Ensemble)
  - Oracle元分类器 (Oracle Meta-Classifier)
- **完整的数据预处理流程**: 包括缺失值处理、异常值检测、特征编码、类别不平衡处理等
- **全面的评估指标**: 包含准确率、精确率、召回率、F1分数、ROC-AUC、Cohen's Kappa、MCC等

## 数据集

项目支持三个数据集：
- **Transaction Dataset**: 银行客户交易数据
- **Telecommunication Dataset**: 电信客户数据
- **Customer Churn Dataset**: 客户流失数据

## 项目结构

```
Customer churn prediction using a novel meta-classifier/
├── data/                           # 数据集文件夹
│   ├── customer_churn.xlsx
│   ├── telecommunication.csv
│   └── transaction.csv
├── src/                           # 源代码文件夹
│   ├── main.py                    # 主程序入口
│   ├── train.py                   # 模型训练模块
│   ├── data_preprocess.py         # 数据预处理模块
│   ├── hyperparameter_optimization.py  # 超参数优化模块
│   ├── utils.py                   # 工具函数
│   └── test.py                    # 测试模块
├── results/                       # 结果文件夹
│   ├── hyperparameter/            # 超参数优化结果
│   ├── metrics/                   # 评估指标结果
│   └── models/                    # 训练好的模型
├── README.md                      # 项目说明文档
├── requirements.txt               # 依赖包列表
├── .gitignore                     # Git忽略文件
└── LICENSE                        # 开源许可证
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 超参数优化

首先需要对各个模型进行超参数优化：

```python
from src.hyperparameter_optimization import HyperparameterOptimization

# 对transaction数据集进行SKB+GridSearch优化
hpo = HyperparameterOptimization('transaction', 'label_one_hot', 'skb_grid')
hpo.save_hyperparameter_params()

# 对transaction数据集进行PFS+BayesSearch优化
hpo = HyperparameterOptimization('transaction', 'label_one_hot', 'pfs_bayes')
hpo.save_hyperparameter_params()
```

### 2. 模型训练

训练不同类型的模型：

```python
from src.train import Train

# 训练单一模型
trainer = Train('transaction', 'label_one_hot', 'single', 'skb_grid')
trainer.train()

# 训练堆叠集成模型
trainer = Train('transaction', 'label_one_hot', 'stack_ensemble', 'skb_grid')
trainer.train()

# 训练Oracle元分类器
trainer = Train('transaction', 'label_one_hot', 'oracle_ensemble', 'skb_grid')
trainer.train()
```

### 3. 运行完整实验

运行主程序进行完整的实验：

```python
from src.main import dataset_main

# 运行customer_churn数据集的完整实验
results = dataset_main('customer_churn', iter_nums=10)
```

## 核心算法

### Oracle元分类器

本项目实现的核心算法是基于Oracle的元分类器，该算法：

1. 训练多个基分类器（决策树、随机森林、XGBoost等）
2. 使用Oracle分类器作为理想情况下的元分类器
3. Oracle分类器能够为每个测试样本选择最优的基分类器
4. 这种方法在理论上能够达到所有集成分类器性能的上限

### 特征选择策略

- **SKB (SelectKBest)**: 使用F统计量进行特征选择
- **PFS (Permutation Feature Importance)**: 使用排列重要性进行特征选择

### 超参数优化

- **GridSearchCV**: 网格搜索交叉验证
- **BayesSearchCV**: 贝叶斯搜索交叉验证

## 评估指标

项目使用多种评估指标来全面评估模型性能：

- **基础指标**: 准确率、精确率、召回率、F1分数
- **ROC指标**: ROC-AUC
- **一致性指标**: Cohen's Kappa、MCC
- **混淆矩阵指标**: TPR、TNR、PPV、NPV、FPR、FNR、FDR、FOR

## 数据预处理

完整的数据预处理流程包括：

1. 删除不必要的列
2. 处理缺失值
3. 删除重复值
4. 异常值检测和处理（IQR方法）
5. 特征编码（标签编码和独热编码）
6. 数据标准化
7. 类别不平衡处理（SMOTEENN）

## 注意事项

- 本项目复现了相关论文的研究成果
- Oracle分类器是理想情况下的分类器，在实际部署中无法使用
- 建议在运行完整实验前先进行超参数优化
- 项目支持多核并行计算以提高效率

## 依赖包

详见 `requirements.txt` 文件。

## 使用说明

本项目仅供学习和研究使用。

## 贡献

欢迎提交Issue和Pull Request来改进本项目。

## 联系方式

如有问题，请通过GitHub Issues联系。
