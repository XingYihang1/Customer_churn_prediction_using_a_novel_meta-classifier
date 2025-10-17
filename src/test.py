'''
测试各个模块是否能够正常工作
'''
from sklearn.tree import DecisionTreeClassifier
from utils import optimization
from data_preprocess import DataPreprocessor
from train import Train

# 测试数据预处理模块
def test_data_preprocess():
    # 检验一下这个类是否能够正常工作
    data_preprocessor = DataPreprocessor(dataset_name="transaction", encoder_type="label_one_hot")
    X_train, X_test, y_train, y_test = data_preprocessor.data_loader()
    print(y_train.value_counts())  # 可以看到已经处理好了类别不平衡
    print('数据预处理模块测试完成')
    print('--------------------------------')
    return X_train, X_test, y_train, y_test

# 测试超参数优化函数
def test_optimization(type, X_train, y_train):
    '''
    测试超参数优化函数:使用一个简单的分类模型和很小的参数空间。
    '''
    print('调优一个简单的模型以测试超参数优化程序是否正常运行......')
    best_model, best_params, selected_features = optimization(DecisionTreeClassifier, "transaction", "label_one_hot", X_train, y_train, type, is_test=True)
    print('超参数优化模块测试完成')
    print('--------------------------------')

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    # 测试数据处理和超参数优化模块
    # X_train, X_test, y_train, y_test = test_data_preprocess()
    # test_optimization('skb_grid', X_train, y_train)
    # test_optimization('pfs_bayes', X_train, y_train)

    # 测试Train类是否能够正常工作
    # train = Train(dataset_name="transaction", encoder_type="label_one_hot", model_type="single", type="skb_grid", save_or_not=False)
    # trained_model_dict, selected_features_dict = train.train()
    # print('单个模型：', trained_model_dict)
    # print('Train类测试完成')
    # print('--------------------------------')

    # train = Train(dataset_name="transaction", encoder_type="label_one_hot", model_type="stack_ensemble", type="skb_grid", save_or_not=False)
    # stack_model, selected_features = train.train()
    # print('堆叠集成模型：', stack_model)
    # print('Train类测试完成')
    # print('--------------------------------')

    # train = Train(dataset_name="transaction", encoder_type="label_one_hot", model_type="oracle_ensemble", type="skb_grid", save_or_not=False)
    # y_pred_final, y_prob_final, _ = train.train()
    # print('Oracle集成模型：', y_pred_final, y_prob_final)
    # print('Train类测试完成')
    # print('--------------------------------')

    # print('所有模块均能正常工作，测试完成')
    # print('--------------------------------')