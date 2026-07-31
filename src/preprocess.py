import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import MinMaxScaler, StandardScaler

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
negative_indicators = ['X8']
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()
normalizations = ['vector', 'MinMax', 'zscore']
weightings = ['IDOCRIW', 'entropy', 'cilos']



# read the normalized data
def read_data(year, normalization):
    data = pd.read_excel(f'../data/processed/{year}_{normalization}_normalized.xlsx')[indicators].to_numpy()
    return data

# transform negative indicators
def transform_negative(data):
    data[negative_indicators] = 1 / data[negative_indicators]
    return data

# vector normalization
def vector_normalization(data):
    column_sums = np.sum(data[indicators] ** 2, axis=0)
    column_sums = column_sums ** 0.5
    data = data / column_sums
    return data

# MinMax normalization
def MinMax_normalization(data):
    scaler = MinMaxScaler()
    data[indicators] = scaler.fit_transform(data[indicators])
    return data

# zscore normalization
def zscore_normalization(data):
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(data[indicators])
    a, b = 0, 1
    z_scores = (z_scores - np.min(z_scores)) / (np.max(z_scores) - np.min(z_scores)) * (b - a) + a
    data[indicators] = z_scores
    return data

# obtain the entropy weight
def obtain_entropy_weight(data):
    column_norm = np.sum(data, axis=0)
    entropy_matrix = data / column_norm
    k = (-1) / math.log(data.shape[0])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            num = entropy_matrix[i][j]
            try:
                entropy_matrix[i][j] = num * math.log(num)
            except ValueError:
                entropy_matrix[i][j] = 0
    entropy = np.sum(entropy_matrix, axis=0)
    entropy = k * entropy
    entropy_weight = (1 - entropy) / (sum(1 - entropy))
    return entropy_weight

# calculate the CILOS weight
def calculate_cilos_weights(data):
    entropy_weights = obtain_entropy_weight(data)
    m, n = data.shape
    cilos_weights = np.zeros(n)
    for j in range(n):
        # remove attribute j and recalculate the weights
        excluded_matrix = np.delete(data, j, axis=1)
        excluded_weights = np.delete(entropy_weights, j)
        excluded_weights = excluded_weights / np.sum(excluded_weights)  # renormalize the weights
        excluded_scores = np.sum(excluded_matrix * excluded_weights, axis=1)

        # the original aggregated score
        original_scores = np.sum(data * entropy_weights, axis=1)

        # calculate the loss caused by attribute j
        loss = np.sum(np.abs(original_scores - excluded_scores))
        cilos_weights[j] = loss
    # normalize the CILOS weight
    cilos_weights = cilos_weights / np.sum(cilos_weights)
    return cilos_weights


# integrate the entropy weight and the cilos weight, generating the IDOCRIW weight
def integrate_weights(data):
    '''integrate the entropy weight and the cilos weight, generating the IDOCRIW weight'''
    entropy_weights = obtain_entropy_weight(data)
    cilos_weights = calculate_cilos_weights(data)
    integrated_weights = (entropy_weights + cilos_weights) / 2  # average
    integrated_weights = integrated_weights / np.sum(integrated_weights)  # normalization
    return integrated_weights

def collect_weights(year, normalization):
    '''collect the three types of weights'''
    data = read_data(year, normalization)
    entropy_weights = obtain_entropy_weight(data)
    cilos_weights = calculate_cilos_weights(data)
    integrated_weights = integrate_weights(data)
    weights = np.vstack([entropy_weights, cilos_weights, integrated_weights])
    weights = pd.DataFrame(weights)
    weights.index = ['entropy', 'cilos', 'IDOCRIW']
    weights.columns = indicators
    # print(f'{year}weights\n', weights)
    weights.to_excel(f'../data/processed/{year}_{normalization}_weights.xlsx')
    return weights


# collect the weighted data for future use
def weighted(year, normalization):
    '''collect the weighted data for future use'''
    weights = collect_weights(year, normalization)
    for weighting in weights.index:
        weight = weights.loc[weighting].to_numpy()
        data = read_data(year, normalization)
        m, n = data.shape
        for i in range(m):
            for j in range(n):
                data[i][j] = data[i][j] * weight[j]
        data = pd.DataFrame(data)
        data.columns = indicators
        print(f'{year}-{normalization}-{weighting}\n', data.head())
        data.insert(0, 'State', states)
        data.to_excel(f'../data/processed/{year}_{normalization}_{weighting}.xlsx', index=False)
    return


for year in years:
    for normalization in normalizations:
        weighted(year, normalization)