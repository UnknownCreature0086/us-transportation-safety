import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata, norm, f_oneway
import statsmodels.api as sm

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()

def obtain_rank(data):
    '''Obtain the rank(Suppose the neagative indicators have already been transformed.)'''
    m, n = data.shape
    maximum = np.max(data, axis=0)
    minimum = np.min(data, axis=0)
    norm = (m - 1) / (maximum - minimum)
    data = 1 + (data - minimum) * norm
    return data

def obtain_RSRw(data):
    m, n = data.shape
    RSRw = (np.sum(data, axis=1)) / (m * n)
    return RSRw

def calculate_rsr(year):
    data = pd.read_excel(f'../data/processed/{year}_IDOCRIW_weighted_data.xlsx')[indicators].to_numpy()
    ranks = obtain_rank(data)
    rsr = obtain_RSRw(data)
    return rsr, ranks


def probit_transformation(rsr):
    """概率单位(Probit)转换"""
    # 计算累计频率 (P = (rank - 0.5)/n)
    sorted_rsr = np.sort(rsr)
    n = len(rsr)
    rank = np.arange(1, n + 1)
    p = (rank - 0.5) / n

    # 计算概率单位Y (标准正态分布的逆累积分布函数)
    y = norm.ppf(p) + 5  # Probit = Φ⁻¹(p) + 5
    return sorted_rsr, p, y


def rsr_regression(sorted_rsr, y):
    """RSR与概率单位Y的线性回归"""
    X = sm.add_constant(y)  # 添加截距项
    model = sm.OLS(sorted_rsr, X)
    results = model.fit()
    return results


def rsr_grouping(rsr, n_groups=3):
    """基于概率单位的分档"""
    # 计算分档阈值 (基于标准正态分布)
    thresholds = norm.ppf(np.linspace(0, 1, n_groups + 1)[1:-1]) + 5

    # 计算对应的RSR分档值
    sorted_rsr, p, y = probit_transformation(rsr)
    regression = rsr_regression(sorted_rsr, y)
    rsr_thresholds = regression.predict(sm.add_constant(thresholds))

    # 分档
    groups = 5 - np.digitize(rsr, rsr_thresholds)  # 组别从1开始
    return groups, rsr_thresholds, regression


def variance_analysis(rsr, groups):
    """方差一致性检验和ANOVA"""

    # Bartlett检验 (方差齐性)
    unique_groups = np.unique(groups)
    group_data = [rsr[groups == g] for g in unique_groups]
    bartlett_stat, bartlett_p = stats.bartlett(*group_data)

    # ANOVA
    f_stat, p_value = f_oneway(*group_data)

    return bartlett_stat, bartlett_p, f_stat, p_value


def RSR(year):
    print(f'\nNow we are processing data of year {year}.\n')
    # 1. 计算RSR
    rsr, ranks = calculate_rsr(year)

    # 2. Probit转换和回归
    sorted_rsr, p, y = probit_transformation(rsr)
    regression = rsr_regression(sorted_rsr, y)
    print("RSR~Probit回归方程:\n")
    print(f"RSR = {regression.params[0]:.4f} + {regression.params[1]:.4f}*Y")
    print(f"R² = {regression.rsquared:.4f}")

    # 3. 分档 (默认3档)
    groups, thresholds, _ = rsr_grouping(rsr, n_groups=5)
    print("\n分档阈值(Probit→RSR):")
    for i, thresh in enumerate(thresholds, 1):
        print(f"档{i}阈值: {thresh:.4f}")

    # 4. 方差分析
    bartlett_stat, bartlett_p, f_stat, p_value = variance_analysis(rsr, groups)
    print("\n方差一致性检验(Bartlett):")
    print(f"统计量 = {bartlett_stat:.4f}, p值 = {bartlett_p:.4f}")
    print("\nANOVA结果:")
    print(f"F统计量 = {f_stat:.4f}, p值 = {p_value:.4f}")

    # 5. 结果整理
    results = pd.DataFrame({
        'State': states,
        'RSR': rsr,
        'Probit(Y)': y[np.argsort(rsr).argsort()],  # 对应原始顺序的Y值
        'Group': groups
    })

    print("\n分档结果预览:")
    print(results)

    return results

for year in years:
    RSR(year).to_excel(f'../data/processed/{year}_RSR_groups.xlsx', index=False)








