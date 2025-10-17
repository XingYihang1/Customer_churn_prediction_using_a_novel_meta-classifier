# 导入依赖包
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split

# 数据预处理类
class DataPreprocessor:
    def __init__(self, dataset_name, encoder_type):
        '''
        初始化数据预处理类
        Args:
            dataset_name: 数据集名称,可选有['transaction', 'telecommunication', 'customer_churn']
            encoder_type: 编码器类型,可选有['label', 'label_one_hot']
        Returns:
            None
        '''
        self.dataset_name = dataset_name
        # 按照数据集名称存数据和数值特征以及类别特征
        if dataset_name == "transaction":
            # 这里使用了相对路径，如果需要使用绝对路径，可以修改为绝对路径
            self.dataset = pd.read_csv(f'../data/{dataset_name}.csv')
            self.numerical_features = ['CreditScore', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Age']
            self.categorical_features = ['Geography', 'Gender']
        elif dataset_name == "telecommunication":
            self.dataset = pd.read_csv(f'../data/{dataset_name}.csv')
            self.numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            self.categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                                        'MultipleLines', 'InternetService', 'OnlineSecurity', 
                                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                        'StreamingTV', 'StreamingMovies', 'Contract', 
                                        'PaperlessBilling', 'PaymentMethod','Churn']
        else:
            self.dataset = pd.read_excel(f'../data/{dataset_name}.xlsx')
            self.numerical_features = ["Age", 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
            self.categorical_features = ['Gender','Location']
        
        self.raw_dataset = self.dataset.copy()  # 存一份原始数据集
        self.encoder_type = encoder_type

    # 删除不必要的列
    def remove_unnecessary_columns(self):
        if self.dataset_name == "transaction":
            self.dataset = self.dataset.drop(columns=['RowNumber','CustomerId', 'Surname'])
        elif self.dataset_name == "telecommunication":
            self.dataset = self.dataset.drop(columns=['customerID'])
        elif self.dataset_name == "customer_churn":
            self.dataset = self.dataset.drop(columns=['CustomerID', 'Name'])
        return self.dataset
    
    # 删除缺失值
    def remove_nan_rows(self):
        '''
        在经历过初步的数据探索后，只有telecomunication数据集有缺失值，需要处理
        '''
        if self.dataset_name != 'telecommunication':
            return self.dataset
        else:
            self.dataset['TotalCharges'] = pd.to_numeric(self.dataset['TotalCharges'], errors='coerce')

            # 使用均值填充缺失值
            self.dataset['TotalCharges'] = self.dataset['TotalCharges'].fillna(self.dataset['TotalCharges'].mean())
            return self.dataset
    
    # 删除重复值
    def remove_duplicated_rows(self):
        return self.dataset.drop_duplicates()
    
    # 删除离群值
    def remove_outliers_iqr(self):
        '''
        使用IQR方法来检测离群值
        '''
        for feature in self.numerical_features:
            Q1 = self.dataset[feature].quantile(0.25)
            Q3 = self.dataset[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.dataset = self.dataset[(self.dataset[feature] > lower_bound) & (self.dataset[feature] < upper_bound)]
        return self.dataset
    
    # 可视化类别特征
    def visualize_categorical_features(self):
        for feature in self.categorical_features:
            num = len(self.dataset[feature].value_counts())  # 每个分类特征有多少个类别
            plt.figure(figsize=(4, (num // 4) + 1))
            sns.countplot(y=feature, data=self.dataset, order=self.dataset[feature].value_counts().index)
            plt.title(f'{feature} Distribution')
            plt.show()

    # 将类别特征编码为数值特征
    def encoder(self):
        le = LabelEncoder()
        if self.encoder_type == 'label':
            for category in self.categorical_features:
                self.dataset[category] = le.fit_transform(self.dataset[category])
        elif self.encoder_type == 'label_one_hot':
            for category in self.categorical_features:
                # 对于二元变量用label编码，对于多元变量用one-hot编码
                if self.dataset[category].nunique() == 2:
                    self.dataset[category] = le.fit_transform(self.dataset[category])
                else:
                    self.dataset = pd.concat([self.dataset, pd.get_dummies(self.dataset[category], dtype=int, prefix=category)], axis=1)
                    self.dataset.drop(category, axis=1, inplace=True)
        return self.dataset
    
    # 标准化数值特征：注意标准化前要区分训练集和测试集，否则会导致数据泄露
    def standardize_data(self, X_train):
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_train_scaled[self.numerical_features] = scaler.fit_transform(X_train[self.numerical_features])
        return X_train_scaled, scaler
    
    # 处理类别不平衡
    def handle_imbalance(self, X_train, y_train):
        '''
        使用SMOTEENN方法来处理类别不平衡
        '''
        smote_enn = SMOTEENN()
        X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

    # 划分训练集和测试集
    def data_loader(self, test_size=0.3, random_state=42):
        '''
        加载数据集
        参数：
            test_size: 测试集比例，默认0.3
            random_state: 随机种子，默认42
        返回：一个经过数据预处理后的四元组(X_train, X_test, y_train, y_test)，注意transaction数据集返回的是处理过类别不平衡后的数据
        '''
        # 数据预处理
        # 步骤：1.删除不必要的列 2.删除缺失值 3.删除重复值 4.删除离群值 5.编码
        func = [self.remove_unnecessary_columns, self.remove_nan_rows, self.remove_duplicated_rows, self.remove_outliers_iqr, self.encoder]
        for func in func:
            self.dataset = func()
        
        # 处理类别不平衡
        if self.dataset_name == "transaction":
            X = self.dataset.drop(columns=['Exited'])
            y = self.dataset['Exited']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # 标准化数值特征
            X_train, scaler = self.standardize_data(X_train)
            X_test_scaled = X_test.copy()
            X_test_scaled[self.numerical_features] = scaler.transform(X_test[self.numerical_features])

            X_resampled, y_resampled = self.handle_imbalance(X_train, y_train)
            return X_resampled, X_test_scaled, y_resampled, y_test
        else:
            X = self.dataset.drop(columns=['Churn'])
            y = self.dataset['Churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # 标准化数值特征
            X_train, scaler = self.standardize_data(X_train)
            X_test_scaled = X_test.copy()
            X_test_scaled[self.numerical_features] = scaler.transform(X_test[self.numerical_features])

            return X_train, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    pass
