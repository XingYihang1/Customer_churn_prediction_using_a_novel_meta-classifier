'''
本模块提供超参数优化和特征选择的工具函数和类
'''

'''
特征选择和超参数优化的执行步骤：
两种方案进行比较：
第一个方案：SKB + GS
SelectKBest
+
GridSearchCV

第二个方案：PFS + BS
Permutation Feature Importance (PFS)
+
BayesSearchCV
'''

# 导入依赖包
from datetime import datetime
import time
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from skopt.space import Integer, Real, Categorical

# 导入自己的数据预处理类
from data_preprocess import DataPreprocessor

# 超参数优化和特征选择类
class FS_and_HPO:
    def __init__(self, model, X_train, y_train, n_splits=5, scoring='accuracy'):
        '''
        初始化
        参数：
            model: 机器学习模型，本项目中可供选择的模型有DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, 
                                                   AdaBoostClassifier, ExtraTreesClassifier
            X_train: 训练集特征
            y_train: 训练集标签
            n_splits: 交叉验证的折数,默认为5
            scoring: 评分标准,默认为accuracy
        '''
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.n_splits = n_splits
        self.scoring = scoring
        self.best_model = None
        self.best_params = None
        self.selected_features = None

    # 在这一步中同时进行selectKBest的优化和模型超参数的优化
    def grid_search_cv(self, param_grid):
        '''
        SKB + 网格搜索超参数优化函数(交叉验证策略为StratifiedKFold)
        我们在这个函数中同时进行selectKBest的优化和模型超参数的优化，selectKBest的评分函数为f_classif。
        参数：
            param_grid: 与模型对应的超参数网格
        返回：
            best_params: 模型最优超参数
            best_model: 最优模型
            best_selected_features: 选择的特征
        '''
        print(f"\n{'='*60}")
        print(f"开始SKB + GridSearchCV优化 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模型: {self.model.__name__}")
        print(f"参数网格大小: {len(param_grid)} 个参数组合")
        print(f"{'='*60}")

        # 开始时间
        start_time = time.time()

        pipeline = Pipeline([
            ('selector', SelectKBest(score_func=f_classif)),
            ('model', self.model(random_state=123))
        ])
        grid_search = GridSearchCV(pipeline, param_grid, 
                                cv=StratifiedKFold(n_splits=self.n_splits, random_state=123, shuffle=True), 
                                scoring=self.scoring,
                                n_jobs=-1,
                                verbose=0)
        
        print('正在执行网格搜索.........')
        grid_search.fit(self.X_train, self.y_train)

        # 总耗时
        total_time = time.time() - start_time
        print(f"SKB + GridSearchCV优化完成 -- 总耗时: {total_time:.2f}秒")
        print(f"{'='*60}")

        # 获取选择的特征
        best_selected_features =self.X_train.columns[grid_search.best_estimator_.named_steps['selector'].get_support()].tolist()

        # 获取最优模型
        best_model = grid_search.best_estimator_['model']

        # 获取模型的最优超参数（只获取在参数网格中搜索的参数）
        best_params = grid_search.best_params_
        # 移除selector相关的参数，只保留模型参数
        best_params = {k.replace('model__', ''): v for k, v in best_params.items() if k.startswith('model__')}

        # 输出最优得分
        print(f"SKB + GridSearchCV最优得分: {grid_search.best_score_}")
        return best_params, best_model, best_selected_features
    
    # 在这一步中同时进行PFS的优化和模型超参数的优化
    def bayes_search_cv(self, param_grid, k_range = (5,10)):
        '''
        PFS + 贝叶斯搜索超参数优化函数(交叉验证策略为StratifiedKFold)
        参数：
            param_grid: 与模型对应的超参数网格
            k_range: 特征个数范围
        返回：
            best_params: 模型最优超参数
            best_model: 最优模型
            best_selected_features: 选择的特征
        '''
        print(f"\n{'='*60}")
        print(f"开始PFS + BayesSearchCV优化 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模型: {self.model.__name__}")
        print(f"参数网格大小: {len(param_grid)} 个参数组合")
        print(f"{'='*60}")

        # 为了适应PFS的特点，我们训练一个简单模型(不调参)
        temp_model = self.model(random_state=123)
        temp_model.fit(self.X_train, self.y_train)

        # 计算特征重要性
        perm_importance = permutation_importance(temp_model, self.X_train, self.y_train, random_state=42)
        feature_importance = pd.DataFrame({
            'feature':self.X_train.columns, 
            'importance':perm_importance.importances_mean,
            'std':perm_importance.importances_std}).sort_values(by='importance', ascending=False)
        
        # 开始时间
        start_time = time.time()

        print('正在执行PFS + BayesSearchCV优化.........')
        bestscore = -np.inf
        # 最优模型
        best_model = None
        # 最优超参数
        best_params = None
        # 最优选择的特征
        best_selected_features = None

        for k in range(k_range[0], k_range[1]):
            selected_features = feature_importance.head(k)['feature'].to_list()
            X_train_selected = self.X_train[selected_features]

            # 贝叶斯搜索
            bayes_search = BayesSearchCV(self.model(random_state=123), param_grid, 
                                        cv=StratifiedKFold(n_splits=self.n_splits, random_state=123, shuffle=True), 
                                        scoring=self.scoring,
                                        n_jobs=-1,
                                        verbose=0)
            bayes_search.fit(X_train_selected, self.y_train)

            if bayes_search.best_score_ > bestscore:
                bestscore = bayes_search.best_score_
                best_model = bayes_search.best_estimator_
                best_params = bayes_search.best_params_
                best_selected_features = selected_features
        
        # 总耗时
        total_time = time.time() - start_time
        print(f"PFS + BayesSearchCV优化完成 -- 总耗时: {total_time:.2f}秒")
        print(f"{'='*60}")
        
        # 输出最优得分
        print(f"PFS + BayesSearchCV最优得分: {bestscore}")
        return best_model, best_params, best_selected_features

    # 根据不同的方法进行组合超参数优化
    def combine_search_cv(self, param_grid, type, **kwargs):
        '''
        结合网格搜索和贝叶斯搜索的超参数优化函数
        参数：
            param_grid: 与模型对应的超参数网格
            type: 优化方法，可选有'skb_grid', 'pfs_bayes'
            **kwargs: 对应每个方法的参数，如k_range等
        '''
        if type == 'skb_grid':
            best_params, best_model, best_selected_features = self.grid_search_cv(param_grid)
        elif type == 'pfs_bayes':
            best_model, best_params, best_selected_features = self.bayes_search_cv(param_grid, **kwargs)
        self.best_model = best_model
        self.best_params = best_params
        self.selected_features = best_selected_features

    
    def get_parm_grid(self, type):
        '''
        获取模型对应的超参数网格
        参数：
            type: 优化方法，可选有'skb_grid', 'pfs_bayes'
        返回：
            param_grid: 与模型对应的超参数网格
        '''
        # 保持网格搜索和贝叶斯搜索的超参数一致，分别独立设置
        if self.model == DecisionTreeClassifier:
            if type == 'skb_grid':
                param_grid = {
                    'selector__k': [5, 10],
                    'model__criterion': ['gini', 'entropy'],
                    'model__max_depth': [1, 5, 10, 15, None],
                    'model__max_features': ['sqrt', 'log2'],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__min_impurity_decrease': [0.0, 0.01, 0.05]
                }
            elif type == 'pfs_bayes':
                param_grid = {
                    'criterion': Categorical(['gini', 'entropy']),
                    'max_depth': Categorical([1, 5, 10, 15, None]),
                    'max_features': Categorical(['sqrt', 'log2']),
                    'min_samples_split': Integer(2, 10),
                    'min_samples_leaf': Integer(1, 4),
                    'min_impurity_decrease': Real(0.0, 0.05),
                }
        elif self.model == RandomForestClassifier:
            if type == 'skb_grid':
                param_grid = {
                    'selector__k': [5, 10],
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [1, 5, 10, 15, None],
                    'model__max_features': ['sqrt', 'log2'],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__min_impurity_decrease': [0.0, 0.01, 0.05]
                }
            elif type == 'pfs_bayes':
                param_grid = {
                    'n_estimators': Integer(50, 200),
                    'max_depth': Categorical([1, 5, 10, 15, None]),
                    'max_features': Categorical(['sqrt', 'log2']),
                    'min_samples_split': Integer(2, 10),
                    'min_samples_leaf': Integer(1, 4),
                    'min_impurity_decrease': Real(0.0, 0.05),
                }
        elif self.model == XGBClassifier:
            if type == 'skb_grid':
                param_grid = {
                    'selector__k': [5, 10],
                    'model__max_depth': [3, 6, 9, 12],
                    'model__n_estimators': [50, 100, 200],
                    'model__min_child_weight': [1, 3, 5],
                    'model__learning_rate': [0.05, 0.1, 0.2]
                }
            elif type == 'pfs_bayes':
                param_grid = {
                    'max_depth': Integer(3, 12),
                    'n_estimators': Integer(50, 200),
                    'min_child_weight': Integer(1, 5),
                    'learning_rate': Real(0.05, 0.2),
                }
        elif self.model == AdaBoostClassifier:
            if type == 'skb_grid':
                param_grid = {
                    'selector__k': [5, 10],
                    'model__n_estimators': [50, 100, 200],
                    'model__algorithm': ['SAMME', 'SAMME.R'],
                    'model__learning_rate': [0.05, 0.5, 1.0]
                }
            elif type == 'pfs_bayes':
                param_grid = {
                    'n_estimators': Integer(50, 200),
                    'algorithm': Categorical(['SAMME']),
                    'learning_rate': Real(0.05, 1.0),
                }
        elif self.model == ExtraTreesClassifier:
            if type == 'skb_grid':
                param_grid = {
                    'selector__k': [5, 10],
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [1, 5, 10, 15, None],
                    'model__max_features': ['sqrt', 'log2'],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__min_impurity_decrease': [0.0, 0.01, 0.05]
                }
            elif type == 'pfs_bayes':
                param_grid = {
                    'n_estimators': Integer(50, 200),
                    'max_depth': Categorical([1, 5, 10, 15, None]),
                    'max_features': Categorical(['sqrt', 'log2']),
                    'min_samples_split': Integer(2, 10),
                    'min_samples_leaf': Integer(1, 4),
                    'min_impurity_decrease': Real(0.0, 0.05),
                }
        return param_grid

    def get_test_param_grid(self, type):
        '''
        获取测试模型的超参数网格:测试模型为DecisionTreeClassifier
        参数:
            type: 优化方法，可选有'skb_grid', 'pfs_bayes'
        返回：
            param_grid: 与模型对应的超参数网格
        '''
        if type == 'skb_grid':
            param_grid = {'selector__k': [2,4], 'model__max_depth': [1, 5]}
        if type == 'pfs_bayes':
            param_grid = {'min_samples_split': Integer(2, 5), 'max_depth': Integer(1, 5)}

        return param_grid
    
    # # Permutation Feature Selection特征重要性排序
    # def select_features_pfs(self, method = None, **kwargs):
    #     '''
    #     pfs特征选择函数
    #     参数：
    #         method: 特征选择方法，可选有'threshold','top_k','percentile','stable', 默认为None。
    #         **kwargs: 对应每个方法的参数
    #     返回：
    #         method不为None时，返回(特征选择后的训练数据, 被选择的特征, 特征重要性排序）；
    #         method为None时，返回特征重要性排序。
    #     '''
    #     # 计算特征重要性
    #     perm_importance = permutation_importance(self.best_model, self.X_train, self.y_train, random_state=42)
    #     feature_importance = pd.DataFrame({'feature':self.X_train.columns, 
    #                                         'importance':perm_importance.importances_mean,
    #                                         'std':perm_importance.importances_std}).sort_values(by='importance', ascending=False)
    #     # 根据method选择特征
    #     if method == 'threshold':
    #         threshold = kwargs.get('threshold', 0.01)
    #         X_train_selected = feature_importance[feature_importance['importance'] > threshold]
    #         feature_selected = X_train_selected['feature'].tolist()
    #         return X_train_selected, feature_selected, feature_importance
    #     elif method == 'top_k':
    #         k = kwargs.get('k', 10)
    #         X_train_selected = feature_importance.head(k)
    #         feature_selected = X_train_selected['feature'].tolist()
    #         return X_train_selected, feature_selected, feature_importance
    #     elif method == 'percentile':
    #         percentile = kwargs.get('percentile', 0.75)
    #         threshold = feature_importance['importance'].quantile(percentile)
    #         X_train_selected = feature_importance[feature_importance['importance'] > threshold]
    #         feature_selected = X_train_selected['feature'].tolist()
    #         return X_train_selected, feature_selected, feature_importance
    #     elif method == 'stable':
    #         mean_importance = feature_importance['importance'].mean()
    #         threshold = kwargs.get('threshold', 0.01)
    #         X_train_selected = feature_importance[(feature_importance['std'] < threshold) & 
    #                                             (feature_importance['importance'] > mean_importance)]  # 标准差小于阈值且重要性大于均值的特征
    #         feature_selected = X_train_selected['feature'].tolist()
    #         return X_train_selected, feature_selected, feature_importance
    #     else:
    #         return feature_importance


def optimization(model, dataset_name, encoder_type, X_train, y_train, type,is_test=False):
    '''
    超参数优化函数
    参数：
        model: 机器学习模型，本项目中可供选择的模型有DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, 
                                                   AdaBoostClassifier, ExtraTreesClassifier
        dataset_name: 数据集名称
        encoder_type: 编码器类型，可选有'label', 'label_one_hot'
        X_train: 训练集特征
        y_train: 训练集标签
        type: 优化方法，可选有'skb_grid', 'pfs_bayes'
        is_test: 是否为测试，默认为False
    返回：
        best_model: 最优模型
        best_params: 最优超参数
        selected_features: 选择的特征
    '''
    print(f"\n{'='*100}")
    print(f"开始超参数优化流程 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {model.__name__}")
    print(f"数据集: {dataset_name}")
    print(f"编码器: {encoder_type}")
    print(f"{'='*100}")

    # 超参数优化
    print("\n>>> 开始超参数优化...")
    fs_and_hpo = FS_and_HPO(model, X_train, y_train)
    # 获取超参数网格
    if is_test:
        param_grid = fs_and_hpo.get_test_param_grid(type)
    else:
        param_grid = fs_and_hpo.get_parm_grid(type)
    fs_and_hpo.combine_search_cv(param_grid, type)
    
    print(f"\n{'='*100}")
    print(f"超参数优化流程完成")
    print(f"最优模型: {fs_and_hpo.best_model}")
    print(f"最优参数: {fs_and_hpo.best_params}")
    print(f"选择特征: {fs_and_hpo.selected_features}")
    print(f"{'='*100}")

    return fs_and_hpo.best_model, fs_and_hpo.best_params, fs_and_hpo.selected_features

if __name__ == "__main__":
    pass