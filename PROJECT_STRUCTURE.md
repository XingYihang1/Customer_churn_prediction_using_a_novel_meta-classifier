# 项目结构详细说明

## 目录结构

```
Customer churn prediction using a novel meta-classifier/
├── data/                                    # 数据集文件夹
│   ├── customer_churn.xlsx                 # 客户流失数据集
│   ├── telecommunication.csv               # 电信客户数据集
│   └── transaction.csv                     # 银行交易数据集
├── src/                                    # 源代码文件夹
│   ├── main.py                            # 主程序入口
│   ├── train.py                           # 模型训练模块
│   ├── data_preprocess.py                 # 数据预处理模块
│   ├── hyperparameter_optimization.py     # 超参数优化模块
│   ├── utils.py                           # 工具函数模块
│   └── test.py                            # 测试模块
├── results/                               # 结果文件夹
│   ├── hyperparameter/                    # 超参数优化结果
│   ├── metrics/                           # 评估指标结果
│   └── models/                            # 训练好的模型
├── README.md                              # 项目说明文档
├── requirements.txt                       # 依赖包列表
├── .gitignore                            # Git忽略文件
└── PROJECT_STRUCTURE.md                  # 项目结构说明文档（本文件）
```

## 文件详细说明

### 数据文件 (data/)

- **customer_churn.xlsx**: 客户流失数据集，包含客户特征和流失标签
- **telecommunication.csv**: 电信客户数据集，包含电信服务相关特征
- **transaction.csv**: 银行交易数据集，包含银行客户交易相关特征

### 源代码文件 (src/)

#### main.py
- **功能**: 主程序入口，运行完整的实验流程
- **主要类/函数**:
  - `caculate_metrics()`: 计算各种评估指标
  - `dataset_main()`: 运行指定数据集的完整实验
- **使用方式**: 直接运行或导入调用

#### train.py
- **功能**: 模型训练和集成学习
- **主要类**:
  - `Train`: 训练类，支持三种模型类型
    - 单一模型训练
    - 堆叠集成模型训练
    - Oracle元分类器训练
- **主要方法**:
  - `train_and_save_single_model()`: 训练单一模型
  - `train_and_save_stack_ensemble_model()`: 训练堆叠集成模型
  - `train_and_save_oracle_ensemble_model()`: 训练Oracle元分类器
  - `predict()`: 模型预测

#### data_preprocess.py
- **功能**: 数据预处理
- **主要类**:
  - `DataPreprocessor`: 数据预处理类
- **主要方法**:
  - `remove_unnecessary_columns()`: 删除不必要的列
  - `remove_nan_rows()`: 处理缺失值
  - `remove_duplicated_rows()`: 删除重复值
  - `remove_outliers_iqr()`: 使用IQR方法检测异常值
  - `encoder()`: 特征编码（标签编码和独热编码）
  - `standardize_data()`: 数据标准化
  - `handle_imbalance()`: 处理类别不平衡（SMOTEENN）
  - `data_loader()`: 完整的数据加载流程

#### hyperparameter_optimization.py
- **功能**: 超参数优化和特征选择
- **主要类**:
  - `HyperparameterOptimization`: 超参数优化主类
  - `FS_and_HPO`: 特征选择和超参数优化组合类
- **主要方法**:
  - `grid_search_cv()`: SKB + 网格搜索优化
  - `bayes_search_cv()`: PFS + 贝叶斯搜索优化
  - `get_parm_grid()`: 获取模型对应的超参数网格
  - `save_hyperparameter_params()`: 保存超参数和特征选择结果
  - `load_hyperparameter_params()`: 加载超参数和特征选择结果

#### utils.py
- **功能**: 工具函数
- **主要函数**:
  - `optimization()`: 超参数优化函数

#### test.py
- **功能**: 测试各个模块功能
- **主要函数**:
  - `test_data_preprocess()`: 测试数据预处理模块
  - `test_optimization()`: 测试超参数优化模块

### 结果文件 (results/)

#### hyperparameter/
- 存储超参数优化的结果
- 文件格式: `{dataset_name}_{optimization_type}_hyperparameter_params.pkl`
- 文件格式: `{dataset_name}_{optimization_type}_selected_features.pkl`

#### metrics/
- 存储模型评估指标结果
- 文件格式: `{dataset_name}_metrics.csv`

#### models/
- 存储训练好的模型
- 文件格式: `{dataset_name}_{optimization_type}_{model_name}.pkl`

## 工作流程

### 1. 数据预处理阶段
1. 加载数据集
2. 删除不必要的列
3. 处理缺失值
4. 删除重复值
5. 异常值检测和处理
6. 特征编码
7. 数据标准化
8. 类别不平衡处理

### 2. 超参数优化阶段
1. 选择优化方法（SKB+GridSearch 或 PFS+BayesSearch）
2. 对每个模型进行超参数优化
3. 保存最优超参数和选择的特征

### 3. 模型训练阶段
1. 加载最优超参数和选择的特征
2. 训练单一模型
3. 训练堆叠集成模型
4. 训练Oracle元分类器

### 4. 模型评估阶段
1. 计算各种评估指标
2. 保存评估结果
3. 生成性能报告

## 支持的模型类型

### 单一模型
- DecisionTreeClassifier
- RandomForestClassifier
- XGBClassifier
- AdaBoostClassifier
- ExtraTreesClassifier

### 集成模型
- StackingClassifier (堆叠集成)
- Oracle (Oracle元分类器)

## 支持的优化方法

### 特征选择 + 超参数优化
1. **SKB + GridSearchCV**: SelectKBest + 网格搜索
2. **PFS + BayesSearchCV**: Permutation Feature Importance + 贝叶斯搜索

## 评估指标

### 基础指标
- Accuracy (准确率)
- Precision (精确率)
- Recall (召回率)
- F1-score (F1分数)

### ROC指标
- ROC-AUC

### 一致性指标
- Cohen's Kappa
- MCC (Matthews Correlation Coefficient)

### 混淆矩阵指标
- TPR (True Positive Rate)
- TNR (True Negative Rate)
- PPV (Positive Predictive Value)
- NPV (Negative Predictive Value)
- FPR (False Positive Rate)
- FNR (False Negative Rate)
- FDR (False Discovery Rate)
- FOR (False Omission Rate)

## 使用建议

1. **首次运行**: 建议先运行 `test.py` 确保所有模块正常工作
2. **超参数优化**: 在训练模型前先进行超参数优化
3. **内存管理**: 对于大数据集，建议适当调整批处理大小
4. **并行计算**: 项目支持多核并行，可以充分利用多核CPU
5. **结果保存**: 所有结果都会自动保存到 `results/` 目录下
