'''
本模块为训练模块，训练各个模型，并进行集成学习
'''
# 导入依赖包
import pickle
from re import X
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from deslib.static.oracle import Oracle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
import numpy as np
# 导入自己的模块
from hyperparameter_optimization import HyperparameterOptimization
from data_preprocess import DataPreprocessor

class Train:
    def __init__(self, dataset_name, encoder_type, model_type, type, save_or_not=True):
        '''
        初始化训练类
        参数：
            dataset_name: 数据集名称
            encoder_type: 编码器类型
            model_type: 模型类型: single, stack_ensemble, oracle_ensemble
            type: 训练类型: skb_grid, pfs_bayes
            save_or_not: 是否保存模型，默认为True
        '''
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.encoder_type = encoder_type
        self.type = type
        self.save_or_not = save_or_not

        # 加载数据集
        data_preprocessor = DataPreprocessor(self.dataset_name, self.encoder_type)
        self.X_train, self.X_test, self.y_train, self.y_test = data_preprocessor.data_loader()

    def train_and_save_single_model(self):
        '''
        训练各个单一模型，并保存训练好的模型
        返回：
        None: 如果保存模型
        (trained_model_dict, selected_features_dict): 如果不保存模型
        '''
        hpo = HyperparameterOptimization(self.dataset_name, self.encoder_type, self.type)
        trained_model_dict = {}

        best_params_dict, selected_features_dict = hpo.load_hyperparameter_params()

        if best_params_dict is None:
            print(f"未找到最优超参数和选择的特征，请先进行超参数优化")
            return None
        else:
            for model in [DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, AdaBoostClassifier, ExtraTreesClassifier]:
                model_name = model.__name__
                print(f"\n{'='*60}")
                print(f"正在训练{self.dataset_name}的{model_name}模型")
                print(f"{'='*60}")
                params = best_params_dict[model_name]
                selected_features = selected_features_dict[model_name]
                X_train = self.X_train[selected_features]
                model = model(**params)
                model.fit(X_train, self.y_train)
                trained_model_dict[model_name] = model
        print(f"\n{'='*60}")
        print(f"训练完成")
        print(f"{'='*60}")

        if self.save_or_not:
            for model_name, model in trained_model_dict.items():
                with open(f'../results/models/{self.dataset_name}_{self.type}_{model_name}.pkl', 'wb') as f:
                    pickle.dump(model, f)
            print(f"训练好的模型保存完成")
            print(f"{'='*60}")
            return None
        else:
            return trained_model_dict, selected_features_dict

    def train_and_save_stack_ensemble_model(self):
        '''
        训练和保存堆叠集成模型
        返回：
        None: 如果保存模型
        (stack_model, selected_features): 如果不保存模型
        '''
        hpo = HyperparameterOptimization(self.dataset_name, self.encoder_type, self.type)

        best_params_dict, selected_features_dict = hpo.load_hyperparameter_params()

        # 取各个模型的最优特征的并集作为最终的特征
        selected_features = set()
        for model in [DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, AdaBoostClassifier, ExtraTreesClassifier]:
            selected_features_ = selected_features_dict[model.__name__]
            selected_features.update(selected_features_)
        selected_features = list(selected_features)
        X_train = self.X_train[selected_features]

        base_models = []

        for model in [DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, AdaBoostClassifier, ExtraTreesClassifier]:
            model_name = model.__name__
            model = model(**best_params_dict[model_name])
            base_models.append((model_name, model))
        
        print('开始训练堆叠集成模型......')
        stack_model = StackingClassifier(estimators=base_models, final_estimator = MLPClassifier())
        stack_model.fit(X_train, self.y_train)
        print(f"\n{'='*60}")
        print(f"训练完成")
        print(f"{'='*60}")

        if self.save_or_not:
            # 保存训练好的模型
            with open(f'../results/models/{self.dataset_name}_{self.type}_stack_ensemble.pkl', 'wb') as f:
                pickle.dump(stack_model, f)
            print(f"训练好的模型保存完成")
            print(f"{'='*60}")
            return None
        else:
            return stack_model, selected_features

    # 基于Oracle的新元分类器:本项目的核心算法
    def train_and_save_oracle_ensemble_model(self):
        '''
        训练和保存基于Oracle的新元分类器模型
        注意该元分类器模型保存的不是模型本身，而是预测标签，这点与上面的方法有所不同
        返回：
        None: 如果保存模型
        (y_pred_final, y_prob_final, selected_features): 如果不保存模型
        '''
        hpo = HyperparameterOptimization(self.dataset_name, self.encoder_type, self.type)

        best_params_dict, selected_features_dict = hpo.load_hyperparameter_params()

        # 取各个模型的最优特征的并集作为最终的特征
        selected_features = set()
        for model in [DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, AdaBoostClassifier, ExtraTreesClassifier]:
            selected_features_ = selected_features_dict[model.__name__]
            selected_features.update(selected_features_)
        selected_features = list(selected_features)
        X_train = self.X_train[selected_features]
        X_test = self.X_test[selected_features]

        # y_train = self.y_train
        # 划分训练集和验证集，比例为8:2
        X_train, X_val, y_train, y_val = train_test_split(X_train, self.y_train, test_size=0.2, random_state=42)

        print('开始训练各个基分类器......')
        # 训练各个基分类器
        base_models = {}
        for model in [DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, AdaBoostClassifier, ExtraTreesClassifier]:
            model_name = model.__name__
            model = model(**best_params_dict[model_name])
            model.fit(X_train, y_train)
            base_models[model_name] = model
        print('各个基分类器训练完成')
        print('--------------------------------')

        # 获取验证集上每个样本的后验概率, 便于后面计算delta
        vol_prob_dict = {}
        vol_pred_dict = {}
        for name, model in base_models.items():
            vol_prob_dict[name] = model.predict_proba(X_val)
            vol_pred_dict[name] = model.predict(X_val)
        vol_correct_dict = {}  # 验证集上每个样本是否被正确分类，正确为1，错误为0
        for name, pred in vol_pred_dict.items():
            vol_correct_dict[name] = (pred == y_val).astype(int)
        print('验证集上每个样本的预测结果和正确标签计算完成')
        print('--------------------------------')

        # 获取一个KDTree近邻树用来查找最相似的K个样本
        K = 200
        kdtree = KDTree(X_val)
        print('KDTree近邻树构建完成')
        print('--------------------------------')

        # 聚合各个基分类器模型，这一步是关键
        n_test = X_test.shape[0]
        classes = np.unique(self.y_test)
        # 将类别和标签对应上,方便后续查找对齐每个模型给出的后验概率
        class_to_idx = {label: idx for idx, label in enumerate(classes)}

        print('开始计算各个测试集样本的最终预测概率......')
        # 对于每个测试集样本，找到最相似的K个样本，计算delta(公式在论文中给出)
        y_pred_final = np.zeros(n_test)
        y_prob_final = np.zeros((n_test, len(classes)))
        for j in range(n_test):
            x_j = X_test.iloc[j].values.reshape(1, -1) # 第j个测试样本
            # 找到最相似的K个样本
            distances, indices = kdtree.query(x_j, k=K)
            indices = indices[0]
            # 计算每个分类器在K个相似样本上的局部精确率delta(其实就是后验概率取平均)
            deltas = {}

            # # 按照局部精确率计算delta
            # for name, pred in vol_pred_dict.items():
            #     delta = vol_correct_dict[name].iloc[indices].mean()
            #     deltas[name] = delta
            
            # ------------------------------------------------------------------------------------------------
            true_labels = y_val.iloc[indices]
            for name, model in base_models.items():
                probs_for_true = []
                for idx, true_label in zip(indices, true_labels):
                    probs = vol_prob_dict[name][idx]  # (预测为0的概率, 预测为1的概率)
                    pos = np.where(model.classes_ == true_label)[0][0] # 真实标签在模型中的索引
                    probs_for_true.append(probs[pos])
                delta = np.mean(probs_for_true)
                deltas[name] = delta
            # ------------------------------------------------------------------------------------------------
            
            # 选择delta最大的分类器
            max_delta_model = max(deltas, key=deltas.get)
            y_prob_final[j,:] = base_models[max_delta_model].predict_proba(x_j)[0]
            y_pred_final[j] = base_models[max_delta_model].predict(x_j)

            # ------------------------------------------------------------------------------------------------
            # # 按照delta计算每个基分类器的预测权重
            # weights = {}
            # if sum(deltas.values()) == 0:  # 如果delta都为0，则每个基分类器的权重都为1/n
            #     weights = {name: 1 / len(deltas) for name in deltas.keys()}
            # else:
            #     weights = {name: deltas[name] / sum(deltas.values()) for name in deltas.keys()}
                
            # # 计算该测试样本的最终预测概率
            # final_probs = np.zeros(len(classes))
            # for name, model in base_models.items():
            #     probs = model.predict_proba(x_j)[0]  # (预测为0的概率, 预测为1的概率)，与model.classes_的顺序对应
            #     aligned_probs = np.zeros(len(classes))
            #     for idx, label in enumerate(model.classes_):
            #         aligned_probs[class_to_idx[label]] = probs[idx]  # 将预测概率对齐到class_to_idx的顺序
            #     final_probs += weights[name] * aligned_probs
            # y_prob_final[j,:] = final_probs  # 保存该样本的最终预测概率
            # y_pred_final[j] = classes[np.argmax(final_probs)]  # 保存该样本的最终预测标签
            # ------------------------------------------------------------------------------------------------


        # ------------------------------------------------------------------------------------------------
        # 理想情况下的oracle分类器
        # pool_classifier = []
        # for model_name, model in base_models.items():
        #     pool_classifier.append(model)
        
        # oracle_classifier = Oracle(pool_classifiers = pool_classifier, random_state = 123, n_jobs = -1)
        # oracle_classifier.fit(X_train, y_train)
        # # 注意：Oracle分类器是一种理想情况下的分类器，是所有集成分类器性能的上限，在实际中是无法部署的
        # y_pred_final = oracle_classifier.predict(X_test, self.y_test)
        # y_prob_final = oracle_classifier.predict_proba(X_test, self.y_test)
        # ------------------------------------------------------------------------------------------------
        

        print('最终预测概率计算完成')
        print('--------------------------------')

        if self.save_or_not:
            # 将结果保存下来
            predictions_data = {
                'y_pred_final': y_pred_final,
                'y_prob_final': y_prob_final,
                'y_true': self.y_test,
            }
            with open(f'../results/models/{self.dataset_name}_{self.type}_oracle_ensemble_predictions.pkl', 'wb') as f:
                pickle.dump(predictions_data, f)
            print(f"预测结果保存完成")
            print(f"{'='*60}")
            return None
        else:
            return y_pred_final, y_prob_final, selected_features

    # 训练方法，根据模型类型选择训练和保存模型的方法
    def train(self):
        if self.model_type == 'single':
            return self.train_and_save_single_model()
        elif self.model_type == 'stack_ensemble':
            return self.train_and_save_stack_ensemble_model()
        elif self.model_type == 'oracle_ensemble':
            return self.train_and_save_oracle_ensemble_model()

    def predict(self):
        '''
        预测标签,只适用于不保存模型的情况
        返回：
        若model_type为single，则返回一个元组，第一个元素为预测标签字典，第二个元素为预测概率字典
        若model_type为stack_ensemble，则返回一个元组，第一个元素为预测标签，第二个元素为预测概率
        若model_type为oracle_ensemble，则返回一个元组，第一个元素为预测标签，第二个元素为预测概率
        '''
        if self.save_or_not:
            print(f"请选择不保存模型的模式")
            return None
        else:
            if self.model_type == 'single':
                model_dict, selected_features_dict = self.train_and_save_single_model()
                y_pred_dict = {}
                y_prob_dict = {}
                for model_name, model in model_dict.items():
                    X_test = self.X_test[selected_features_dict[model_name]]
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)
                    y_pred_dict[model_name] = y_pred
                    y_prob_dict[model_name] = y_prob
                return y_pred_dict, y_prob_dict
            elif self.model_type == 'stack_ensemble':
                stack_model, selected_features = self.train_and_save_stack_ensemble_model()
                X_test = self.X_test[selected_features]
                y_pred = stack_model.predict(X_test)
                y_prob = stack_model.predict_proba(X_test)
                return y_pred, y_prob
                
            elif self.model_type == 'oracle_ensemble':
                y_pred_final, y_prob_final, selected_features = self.train_and_save_oracle_ensemble_model()
                return y_pred_final, y_prob_final

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    # 保存模型
    for dataset_name in ['transaction', 'telecommunication', 'customer_churn']:
        for model_type in ['single', 'stack_ensemble', 'oracle_ensemble']:
            for type in ['skb_grid', 'pfs_bayes']:
                print(f"正在训练: {dataset_name}, {model_type}, {type}")
                trainer = Train(dataset_name, 'label_one_hot', model_type, type)
                trainer.train()
                print(f"训练完成: {dataset_name}, {model_type}, {type}")