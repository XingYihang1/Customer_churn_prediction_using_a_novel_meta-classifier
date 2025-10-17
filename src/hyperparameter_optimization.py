'''
超参数优化模块
'''
# 导入依赖包
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

# 导入自己的数据预处理类和超参数优化函数
from data_preprocess import DataPreprocessor
from utils import optimization


# 超参数优化类，用于获取和保存当前数据集下各个模型的最优超参数和选择的特征
class HyperparameterOptimization:
    def __init__(self,dataset_name, encoder_type, type):
        '''
        初始化超参数优化类
        参数：
            dataset_name: 数据集名称
            encoder_type: 编码器类型
            type: 优化方法，可选有'skb_grid', 'pfs_bayes'
        '''
        self.dataset_name = dataset_name
        self.encoder_type = encoder_type
        self.best_params = {}
        self.selected_features = {}
        self.type = type

    def data_preprocess(self):
        '''
        数据预处理
        '''
        print(f"{'='*60}")
        print(f"开始数据预处理")
        print(f"{'='*60}")
        data_preprocessor = DataPreprocessor(self.dataset_name, self.encoder_type)
        X_train, X_test, y_train, y_test = data_preprocessor.data_loader()
        print(f"数据预处理完成------------------")
        return X_train, y_train, X_test, y_test
    
    def get_best_params(self):
        '''
        获取当前数据集下各个模型的最优超参数和选择的特征
        '''
        X_train, y_train, _, _ = self.data_preprocess()
        for model in [DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, AdaBoostClassifier, ExtraTreesClassifier]:
            best_model, best_params, selected_features = optimization(model, self.dataset_name, self.encoder_type, X_train, y_train, self.type)
            self.best_params[model.__name__] = best_params
            self.selected_features[model.__name__] = selected_features
    
    def save_hyperparameter_params(self):
        '''
        保存最优超参数和选择的特征
        '''
        # 获取最优超参数和选择的特征
        self.get_best_params()
        # 保存最优超参数和选择的特征
        with open(f'../results/hyperparameter/{self.dataset_name}_{self.type}_hyperparameter_params.pkl', 'wb') as f:
            pickle.dump(self.best_params, f)
        with open(f'../results/hyperparameter/{self.dataset_name}_{self.type}_selected_features.pkl', 'wb') as f:
            pickle.dump(self.selected_features, f)
    
    def load_hyperparameter_params(self):
        '''
        加载最优超参数和选择的特征
        '''
        with open(f'../results/hyperparameter/{self.dataset_name}_{self.type}_hyperparameter_params.pkl', 'rb') as f:
            self.best_params = pickle.load(f)
        with open(f'../results/hyperparameter/{self.dataset_name}_{self.type}_selected_features.pkl', 'rb') as f:
            self.selected_features = pickle.load(f)
        return self.best_params, self.selected_features
    


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    # 保存各个数据集下各个模型的最优超参数和选择的特征
    # for dataset_name in ['transaction', 'telecommunication', 'bank']:
    #     for type in ['skb_grid', 'pfs_bayes']:
    #         if os.path.exists(f'../results/hyperparameter/{dataset_name}_{type}_hyperparameter_params.pkl'):
    #             continue
    #         else:
    #             hpo = HyperparameterOptimization(dataset_name, 'label_one_hot', type)
    #             hpo.save_hyperparameter_params()
    # hpo = HyperparameterOptimization('customer_churn', 'label_one_hot', 'skb_grid')
    # hpo.save_hyperparameter_params()

    # 测试加载超参数和选择的特征，看看是否能正常加载
    # hpo = HyperparameterOptimization('transaction', 'label_one_hot', 'skb_grid')
    # best_params_dict, selected_features_dict = hpo.load_hyperparameter_params()
    # print(best_params_dict)
    # print(selected_features_dict)
    # 可以正常加载，说明超参数优化模块正常工作