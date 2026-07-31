import pandas as pd
import numpy as np

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()


def VIKOR(year):
    # read the preprocessed data
    data = pd.read_excel(f'../data/processed/{year}_IDOCRIW_weighted_data.xlsx')[indicators].to_numpy()
    m, n = data.shape
    ideal_best = np.max(data, axis=0)
    ideal_worst = np.min(data, axis=0)
    # 计算群体效益 S_i 和个体遗憾 R_i
    S = np.zeros(m)
    R = np.zeros(m)
    for i in range(m):
        S[i] = np.sum((ideal_best - data[i])/ (ideal_best - ideal_worst))
        R[i] = np.max((ideal_best - data[i]) / (ideal_best - ideal_worst))
    # 计算妥协指数 Q_i
    v = 0.5  # 偏好参数，通常取0.5
    S_star = np.min(S)
    S_worst = np.max(S)
    R_star = np.min(R)
    R_worst = np.max(R)
    Q = v * (S - S_star) / (S_worst - S_star) + (1 - v) * (R - R_star) / (R_worst - R_star)
    result = pd.DataFrame({
        'State': states,
        'VIKOR': Q,
    })
    result['Ranking'] = result['VIKOR'].rank(ascending=True, method='min').astype(int)
    print(result)
    return result

for year in years:
    VIKOR(year).to_excel(f'../data/processed/{year}_VIKOR.xlsx', index=False)