import numpy as np
import pandas as pd

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
negative_indicators = ['X8']
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()
psi1, psi2, zeta, delta = 1, 1, 1, 1

def read_weights(year):
    weights = pd.read_excel(f'../data/processed/{year}_weights.xlsx')
    weights.columns = ['method'] + indicators
    weight = weights[weights['method'] == 'IDOCRIW'][indicators].to_numpy()[0]  # 添加[0]获取第一行作为1D数组
    return weight

def DOBI(year):
    # read the normalized data
    data = pd.read_excel(f'../data/processed/{year}_vector_normalized.xlsx')[indicators].to_numpy()

    # read weights
    weights = read_weights(year)

    # 2. calculate value of f
    row_sums = np.sum(data, axis=1, keepdims=True)
    f_values = data / row_sums

    # 3. DOBI function
    n_samples, n_features = data.shape
    Z1 = np.zeros(n_samples)
    Z2 = np.zeros(n_samples)

    for i in range(n_samples):
        # Z1
        numerator = np.sum(data[i, :])
        denominator_terms = []
        for j in range(n_features):
            for k in range(n_features):
                if j != k:
                    wj = weights[j]
                    wk = weights[k]
                    term1 = psi1 * ((1 - f_values[i, j]) / f_values[i, j]) ** zeta
                    term2 = psi2 * ((1 - f_values[i, k]) / f_values[i, k]) ** zeta
                    denominator_terms.append((wj * wk / (1 - wj)) * (term1 + term2))
        Z1[i] = numerator / (1 + (np.sum(denominator_terms) / (psi1 + psi2)) ** (1 / zeta))

        # Z2
        denominator_terms = []
        for j in range(n_features):
            for k in range(n_features):
                if j != k:
                    wj = weights[j]
                    wk = weights[k]
                    term1 = psi1 * (f_values[i, j] / (1 - f_values[i, j])) ** zeta
                    term2 = psi2 * (f_values[i, k] / (1 - f_values[i, k])) ** zeta
                    denominator_terms.append((wj * wk / (1 - wj)) * (term1 + term2))
        Z2[i] = numerator / (1 + (np.sum(denominator_terms) / (psi1 + psi2)) ** (1 / zeta))

    # 4. integrated values
    term1 = 0.5 * ((1 - Z1) / Z1) ** delta
    term2 = 0.5 * ((1 - Z2) / Z2) ** delta
    integrated_values = (Z1 + Z2) / (1 + (term1 + term2) ** (1 / delta))
    # Rank the alternatives
    result = pd.DataFrame({
        'State': states,
        'DOBI': integrated_values
    })
    result['Ranking'] = result['DOBI'].rank(ascending=False, method='min').astype(int)
    print(f'{year}\n', result)
    return result

for year in years:
    DOBI(year).to_excel(f'../data/processed/{year}_DOBI.xlsx', index=False)

