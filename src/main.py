'''
主程序，用于运行整个项目
'''
'''
本项目使用到的评价指标：
混淆矩阵相关指标
TPR (True Positive Rate): 真正率 = TP/(TP+FN)，也叫敏感性(Sensitivity)或召回率(Recall)
TNR (True Negative Rate): 真负率 = TN/(TN+FP)，也叫特异性(Specificity)
PPV (Positive Predictive Value): 阳性预测值 = TP/(TP+FP)，也叫精确率(Precision)
NPV (Negative Predictive Value): 阴性预测值 = TN/(TN+FN)
FPR (False Positive Rate): 假正率 = FP/(FP+TN) = 1 - TNR
FNR (False Negative Rate): 假负率 = FN/(FN+TP) = 1 - TPR
FDR (False Discovery Rate): 假发现率 = FP/(FP+TP) = 1 - PPV
FOR (False Omission Rate): 假遗漏率 = FN/(FN+TN) = 1 - NPV
综合指标
Accuracy (ACC): 准确率 = (TP+TN)/(TP+TN+FP+FN)
Precision: 精确率 = TP/(TP+FP) = PPV
Recall: 召回率 = TP/(TP+FN) = TPR
F1-score: F1分数 = 2×(Precision×Recall)/(Precision+Recall)
ROC-AUC: ROC曲线下面积，衡量分类器整体性能
Cohen's Kappa: 考虑随机一致性的分类一致性指标
MCC (Matthews Correlation Coefficient): 马修斯相关系数，平衡所有混淆矩阵元素的指标
'''

# 导入模块
from train import Train

# 导入依赖包
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef)
import pandas as pd
import numpy as np

def caculate_metrics(model_name, hpo_type, y_pred, y_prob, y_test):
    '''
    计算分类模型评价的相关指标
    参数：
        model_name: 模型名称
        hpo_type: 超参数优化类型，可选有'skb_grid', 'pfs_bayes'
        y_pred: 预测标签
        y_prob: 预测概率
        y_test: 测试集标签
    返回：
        metrics_dict: 包含各种评价指标的字典
    '''
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    # 混淆矩阵相关指标
    TPR = tp / (tp + fn) if (tp + fn) > 0 else 0  # 真正率/召回率
    TNR = tn / (tn + fp) if (tn + fp) > 0 else 0  # 真负率
    PPV = tp / (tp + fp) if (tp + fp) > 0 else 0  # 阳性预测值/精确率
    NPV = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0  # 假正率
    FNR = fn / (fn + tp) if (fn + tp) > 0 else 0  # 假负率
    FDR = fp / (fp + tp) if (fp + tp) > 0 else 0  # 假发现率
    FOR = fn / (fn + tn) if (fn + tn) > 0 else 0  # 假遗漏率

    # 基础指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

    # ROC-AUC指标
    roc_auc = None
    if y_prob is not None:
        try:
            # 确保y_prob是二维数组，对于二分类使用正类概率
            if len(y_prob.shape) == 1:
                roc_auc = roc_auc_score(y_test, y_prob)
            else:
                # 对于二分类，使用正类（索引1）的概率
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        except Exception as e:
            print(f"ROC-AUC计算失败: {e}")
            roc_auc = None

    # 综合指标
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    
    
    # 将所有指标汇总到字典中
    metrics_dict = {
        'model_name': [model_name],
        'hpo_type': [hpo_type],
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1': [f1],
        'roc_auc': [roc_auc],
        'cohen_kappa': [cohen_kappa],
        'TPR': [TPR],
        'TNR': [TNR],
        'PPV': [PPV],
        'NPV': [NPV],
        'FPR': [FPR],
        'FNR': [FNR],
        'FDR': [FDR],
        'FOR': [FOR],
        'MCC': [mcc]
    }

    return metrics_dict
    

def dataset_main(dataset_name, iter_nums=10):
    '''
    主函数，用于运行在某个数据集下执行整个项目,得到各个模型的评价指标
    参数：
        dataset_name: 数据集名称
        iter_nums: 迭代次数,默认10次
    返回：
        当前数据集下每个模型在各个超参数优化类型下的平均评价指标
    '''
    results = []

    for model_type in ['single', 'stack_ensemble', 'oracle_ensemble']:
        for hpo_type in ['skb_grid', 'pfs_bayes']:
            for i in range(iter_nums):
                print(f"{'='*60}")
                print(f'正在运行{model_type}_{hpo_type}的第{i+1}次迭代')
                trainer = Train(dataset_name, 'label_one_hot', model_type, hpo_type, save_or_not=False)
                y_test = trainer.y_test
                if model_type == 'single':
                    y_pred_dict, y_prob_dict = trainer.predict()
                    for model_name, y_pred in y_pred_dict.items():
                        y_prob = y_prob_dict[model_name]
                        metrics_dict = caculate_metrics(model_name, hpo_type, y_pred, y_prob, y_test)
                        results.append(pd.DataFrame(metrics_dict))
                elif model_type == 'stack_ensemble':
                    y_pred, y_prob = trainer.predict()
                    metrics_dict = caculate_metrics('stack_ensemble', hpo_type, y_pred, y_prob, y_test)
                    results.append(pd.DataFrame(metrics_dict))
                elif model_type == 'oracle_ensemble':
                    y_pred, y_prob = trainer.predict()
                    metrics_dict = caculate_metrics('oracle_ensemble', hpo_type, y_pred, y_prob, y_test)
                    results.append(pd.DataFrame(metrics_dict))
                print(f'{model_type}_{hpo_type}的第{i+1}次迭代完成')
                print(f"{'='*60}")
    # 将所有结果合并成一个DataFrame
    all_results_df = pd.concat(results, ignore_index=True)
    
    # 根据model_name, hpo_type进行分组，计算平均值
    avg_df = all_results_df.groupby(['model_name', 'hpo_type']).agg(['mean', 'std']).reset_index()

    # 将结果保存在csv文件中
    avg_df.to_csv(f'../results/metrics/{dataset_name}_metrics.csv', index=False)
    # 返回当前数据集下每个模型在各个超参数优化类型下的平均评价指标
    return avg_df

if __name__ == '__main__':
    import warnings
    import time
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    # transaction数据集
    # print(f"{'='*60}")
    # print(f'正在运行transaction数据集')
    # print(f"{'='*60}")
    
    # start_time = time.time()
    # dataset_name = 'transaction'
    # iter_nums = 10
    # transaction_df =dataset_main(dataset_name, iter_nums)
    # end_time_transaction = time.time()
    
    # print(f"transaction数据集运行时间: {end_time_transaction - start_time}秒")
    # print(f"{'='*60}")
    
    # telecommunication数据集
    # print(f"{'='*60}")
    # print(f'正在运行telecommunication数据集')
    # print(f"{'='*60}")
    # start_time = time.time()
    # dataset_name = 'telecommunication'
    # iter_nums = 10
    # telecommunication_df = dataset_main(dataset_name, iter_nums)
    # end_time_telecommunication = time.time()
    # print(f"telecommunication数据集运行时间: {end_time_telecommunication - start_time}秒")
    # print(f"{'='*60}")
    
    # # customer_churn数据集
    print(f"{'='*60}")
    print(f'正在运行customer_churn数据集')
    print(f"{'='*60}")
    start_time = time.time()
    dataset_name = 'customer_churn'
    iter_nums = 10
    customer_churn_df = dataset_main(dataset_name, iter_nums)
    end_time_customer_churn = time.time()
    print(f"customer_churn数据集运行时间: {end_time_customer_churn - start_time}秒")
    print(f"{'='*60}")
