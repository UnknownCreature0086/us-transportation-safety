import pandas as pd
import numpy as np
import math

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()


# read the normalized data
def read_data(year):
    data = pd.read_excel(f'../data/processed/{year}_vector_normalized.xlsx')[indicators].to_numpy()
    return data

# obtain the entropy weight
def obtain_entropy_weight(year):
    data = read_data(year)
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
def calculate_cilos_weights(year):
    data = read_data(year)
    entropy_weights = obtain_entropy_weight(year)
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
def integrate_weights(year):
    '''integrate the entropy weight and the cilos weight, generating the IDOCRIW weight'''
    entropy_weights = obtain_entropy_weight(year)
    cilos_weights = calculate_cilos_weights(year)
    integrated_weights = (entropy_weights + cilos_weights) / 2  # average
    integrated_weights = integrated_weights / np.sum(integrated_weights)  # normalization
    return integrated_weights

def collect_weights(year):
    '''collect the three types of weihgts'''
    entropy_weights = obtain_entropy_weight(year)
    cilos_weights = calculate_cilos_weights(year)
    integrated_weights = integrate_weights(year)
    weights = np.vstack([entropy_weights, cilos_weights, integrated_weights])
    weights = pd.DataFrame(weights)
    weights.index = ['entropy', 'cilos', 'IDOCRIW']
    weights.columns = indicators
    print(f'{year}weights\n', weights)
    return weights

# collect the weighted data for future use
def weighted(year):
    '''collect the weighted data for future use'''
    weights = collect_weights(year)
    for method in weights.index:
        weight = weights.loc[method].to_numpy()
        data = read_data(year)
        m, n = data.shape
        for i in range(m):
            for j in range(n):
                data[i][j] = data[i][j] * weight[j]
        data = pd.DataFrame(data)
        data.columns = indicators
        data.insert(0, 'State', states)
        print(f'{year} {method} weighted data\n', data)
        data.to_excel(f'../data/processed/{year}_{method}_weighted_data.xlsx', index=False)
    return


for year in years:
    collect_weights(year).to_excel(f'../data/processed/{year}_weights.xlsx')
    weighted(year)