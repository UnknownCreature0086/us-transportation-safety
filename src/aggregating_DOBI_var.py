import numpy as np
import pandas as pd

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
negative_indicators = ['X8']
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()
normalizations = ['vector', 'MinMax', 'zscore']
weightings = ['IDOCRIW', 'entropy', 'cilos']
psi1, psi2, zeta, delta = 1, 1, 1, 1

def read_weights(year, normalization):
    weights = pd.read_excel(f'../data/processed/{year}_{normalization}_weights.xlsx')
    weights.columns = ['method'] + indicators
    weight = weights[weights['method'] == 'IDOCRIW'][indicators].to_numpy()[0]  # 添加[0]获取第一行作为1D数组
    return weight


def DOBI(year, normalization, weighting):
    # read the normalized data
    data = pd.read_excel(f'../data/processed/{year}_{normalization}_{weighting}.xlsx')[indicators].to_numpy()

    # read weights
    weights = read_weights(year, normalization)

    # Calculate f_values with epsilon protection
    epsilon = 1e-10
    row_sums = np.sum(data, axis=1, keepdims=True)
    f_values = data / (row_sums + epsilon)

    # DOBI calculation
    n_samples, n_features = data.shape
    Z1 = np.zeros(n_samples)
    Z2 = np.zeros(n_samples)

    for i in range(n_samples):
        # Z1 calculation
        numerator = np.sum(data[i, :])
        denominator_terms = []
        for j in range(n_features):
            for k in range(n_features):
                if j != k:
                    wj = weights[j]
                    wk = weights[k]
                    denom_j = f_values[i, j] if f_values[i, j] != 0 else epsilon
                    denom_k = f_values[i, k] if f_values[i, k] != 0 else epsilon
                    term1 = psi1 * ((1 - f_values[i, j]) / denom_j) ** zeta
                    term2 = psi2 * ((1 - f_values[i, k]) / denom_k) ** zeta
                    denominator_terms.append((wj * wk / (1 - wj + epsilon)) * (term1 + term2))
        Z1[i] = numerator / (1 + (np.sum(denominator_terms) / (psi1 + psi2 + epsilon)) ** (1 / zeta))

        # Z2 calculation
        denominator_terms = []
        for j in range(n_features):
            for k in range(n_features):
                if j != k:
                    wj = weights[j]
                    wk = weights[k]
                    denom_j = f_values[i, j] if f_values[i, j] != 0 else epsilon
                    denom_k = f_values[i, k] if f_values[i, k] != 0 else epsilon
                    term1 = psi1 * (f_values[i, j] / (1 - f_values[i, j] + epsilon)) ** zeta
                    term2 = psi2 * (f_values[i, k] / (1 - f_values[i, k] + epsilon)) ** zeta
                    denominator_terms.append((wj * wk / (1 - wj + epsilon)) * (term1 + term2))
        Z2[i] = numerator / (1 + (np.sum(denominator_terms) / (psi1 + psi2 + epsilon)) ** (1 / zeta))

    # Integrated values with array-safe operations
    denom_z1 = np.where(Z1 != 0, Z1, epsilon)
    denom_z2 = np.where(Z2 != 0, Z2, epsilon)
    term1 = 0.5 * ((1 - Z1) / denom_z1) ** delta
    term2 = 0.5 * ((1 - Z2) / denom_z2) ** delta
    integrated_values = (Z1 + Z2) / (1 + (term1 + term2) ** (1 / delta))

    # Return results
    result = pd.DataFrame({
        'State': states,
        'DOBI': integrated_values
    })
    result['Ranking'] = result['DOBI'].rank(ascending=False, method='min').astype(int)
    return result

for year in years:
    for normalization in normalizations:
        for weighting in weightings:
            DOBI(year, normalization, weighting).to_excel(f'../data/processed/{year}_{normalization}_{weighting}_DOBI.xlsx', index=False)