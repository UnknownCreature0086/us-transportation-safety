import pandas as pd
import numpy as np
import math

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]


# read the normalized data
def read_data(year):
    data = pd.read_excel(f'../data/processed/{year}_vector_normalized.xlsx')
    return data

# obtain the entropy weight
def obtain_entropy_weight(year):
    data = read_data(year)[indicators]
    data = data.to_numpy()
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
    entropy_weight = pd.DataFrame(entropy_weight).transpose()
    entropy_weight.columns = indicators
    entropy_weight.index = ['entropy']
    return entropy_weight

for year in years:
    obtain_entropy_weight(year).to_excel(f'../data/processed/{year}_entropy_weight.xlsx')