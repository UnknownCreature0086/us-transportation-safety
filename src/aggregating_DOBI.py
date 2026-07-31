import numpy as np
import pandas as pd

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
negative_indicators = ['X8']
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()

class DOBI_Method:
    def __init__(self, weights=None, psi1=1, psi2=1, zeta=1, delta=1):
        """
        初始化DOBI方法
        :param weights: 各特征的权重向量
        :param psi1: Bonferroni函数参数1
        :param psi2: Bonferroni函数参数2
        :param zeta: Dombi运算参数
        :param delta: 综合函数参数
        """
        self.weights = weights
        self.psi1 = psi1
        self.psi2 = psi2
        self.zeta = zeta
        self.delta = delta

    def normalize_matrix(self, X, benefit_attributes=None):
        """
        标准化决策矩阵
        :param X: 原始决策矩阵 (样本数×特征数)
        :param benefit_attributes: 指示哪些特征是效益型(True)还是成本型(False)
        :return: 标准化后的矩阵
        """
        if benefit_attributes is None:
            # 默认为所有特征都是效益型
            benefit_attributes = np.ones(X.shape[1], dtype=bool)

        X_norm = np.zeros_like(X, dtype=float)

        for j in range(X.shape[1]):
            col = X[:, j]
            max_val = np.max(col)

            if benefit_attributes[j]:
                # 效益型特征标准化
                X_norm[:, j] = col / max_val
            else:
                # 成本型特征标准化
                min_val = np.min(col)
                X_norm[:, j] = (min_val / col) + (max_val / max_val) + (min_val / max_val)

        return X_norm

    def calculate_f_values(self, X_norm):
        """
        计算f(θ)值 - 加性归一化
        :param X_norm: 标准化后的矩阵
        :return: f值矩阵
        """
        row_sums = np.sum(X_norm, axis=1, keepdims=True)
        return X_norm / row_sums

    def dombi_bonferroni_avg(self, X_norm, f_values):
        """
        计算DOBI加权平均函数 Z1
        :param X_norm: 标准化矩阵
        :param f_values: f(θ)值矩阵
        :return: Z1值向量
        """
        n_samples, n_features = X_norm.shape
        Z1 = np.zeros(n_samples)

        for i in range(n_samples):
            numerator = np.sum(X_norm[i, :])

            denominator_terms = []
            for j in range(n_features):
                for k in range(n_features):
                    if j != k:
                        wj = self.weights[j]
                        wk = self.weights[k]

                        term1 = self.psi1 * ((1 - f_values[i, j]) / f_values[i, j]) ** self.zeta
                        term2 = self.psi2 * ((1 - f_values[i, k]) / f_values[i, k]) ** self.zeta

                        denominator_term = (wj * wk / (1 - wj)) * (term1 + term2)
                        denominator_terms.append(denominator_term)

            denominator = 1 + (np.sum(denominator_terms) / (self.psi1 + self.psi2)) ** (1 / self.zeta)

            Z1[i] = numerator / denominator

        return Z1

    def dombi_bonferroni_geo(self, X_norm, f_values):
        """
        计算DOBI加权几何函数 Z2
        :param X_norm: 标准化矩阵
        :param f_values: f(θ)值矩阵
        :return: Z2值向量
        """
        n_samples, n_features = X_norm.shape
        Z2 = np.zeros(n_samples)

        for i in range(n_samples):
            numerator = np.sum(X_norm[i, :])

            denominator_terms = []
            for j in range(n_features):
                for k in range(n_features):
                    if j != k:
                        wj = self.weights[j]
                        wk = self.weights[k]

                        term1 = self.psi1 * (f_values[i, j] / (1 - f_values[i, j])) ** self.zeta
                        term2 = self.psi2 * (f_values[i, k] / (1 - f_values[i, k])) ** self.zeta

                        denominator_term = (wj * wk / (1 - wj)) * (term1 + term2)
                        denominator_terms.append(denominator_term)

            denominator = 1 + (np.sum(denominator_terms) / (self.psi1 + self.psi2)) ** (1 / self.zeta)

            Z2[i] = numerator / denominator

        return Z2

    def integrated_dobi_function(self, Z1, Z2):
        """
        计算综合DOBI函数值 ℜ
        :param Z1: DOBI加权平均函数值
        :param Z2: DOBI加权几何函数值
        :return: 综合值向量
        """
        numerator = Z1 + Z2

        term1 = 0.5 * ((1 - Z1) / Z1) ** self.delta
        term2 = 0.5 * ((1 - Z2) / Z2) ** self.delta

        denominator = 1 + (term1 + term2) ** (1 / self.delta)

        return numerator / denominator

    def evaluate(self, X, benefit_attributes=None):
        """
        综合评价过程
        :param X: 原始决策矩阵
        :param benefit_attributes: 特征类型指示
        :return: 排序结果
        """
        # 1. 标准化矩阵
        X_norm = self.normalize_matrix(X, benefit_attributes)

        # 2. 计算f值
        f_values = self.calculate_f_values(X_norm)

        # 3. 计算DOBI函数
        Z1 = self.dombi_bonferroni_avg(X_norm, f_values)
        Z2 = self.dombi_bonferroni_geo(X_norm, f_values)

        # 4. 计算综合值
        integrated_values = self.integrated_dobi_function(Z1, Z2)

        # 5. 排序
        rankings = np.argsort(-integrated_values)  # 降序排列

        results = pd.DataFrame({
            'Sample': np.arange(1, len(integrated_values) + 1),
            'Z1': Z1,
            'Z2': Z2,
            'Integrated_Value': integrated_values,
            'Rank': rankings + 1  # 从1开始的排名
            })

        # return results.sort_values('Rank')
        return results

def read_data(year):
    df = pd.read_excel('../data/raw/US Transport Data_2010-2022.xlsx', sheet_name=f'{year}')
    df.columns = column_names
    df['State'] = states
    return df

def read_weights(year):
    weights = pd.read_excel(f'../data/processed/{year}_weights.xlsx')
    weights.columns = ['method'] + indicators
    weight = weights[weights['method'] == 'IDOCRIW'][indicators].to_numpy()[0]  # 添加[0]获取第一行作为1D数组
    print(f'{year} weight shape:', weight.shape)  # 调试用，确认形状
    print(f'{year} weight values:\n', weight)
    return weight

for year in years:
    data = read_data(year)[indicators].to_numpy()
    weights = read_weights(year)
    # 前7个特征是效益型，后1个是成本型
    benefit_attrs = [True] * 7 + [False] * 1
    # 创建DOBI评估器
    dobi = DOBI_Method(weights=weights, psi1=1, psi2=1, zeta=1, delta=20)
    # 进行评估
    results = dobi.evaluate(data, benefit_attributes=benefit_attrs)
    # 打印前10个样本的结果
    print(f'{year}DOBI综合评价结果(前10个state):\n')
    print(results.head(10))


