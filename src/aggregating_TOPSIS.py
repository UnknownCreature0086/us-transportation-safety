import pandas as pd
import numpy as np

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()

# positive ideal solution and negative ideal solution
def ideal_solution(matrix):
    '''positive ideal solution and negative ideal solution'''
    maximum = np.max(matrix, axis=0)
    minimum = np.min(matrix, axis=0)
    positive_distance = (np.sum((matrix - maximum) ** 2, axis=1)) ** 0.5
    negative_distance = (np.sum((matrix - minimum) ** 2, axis=1)) ** 0.5
    return [positive_distance, negative_distance]


# aggregated index
def ranking_index(matrix):
    '''aggregated index'''
    positive = ideal_solution(matrix)[0]
    negative = ideal_solution(matrix)[1]
    index = negative / (positive + negative)
    return index

def TOPSIS(year):
    '''the main TOPSIS process'''
    # read the preprocessed data
    data = pd.read_excel(f'../data/processed/{year}_IDOCRIW_weighted_data.xlsx')
    index = ranking_index(data[indicators])
    # Rank the alternatives
    result = pd.DataFrame({
        'State': states,
        'TOPSIS': index
    })
    result['Ranking'] = result['TOPSIS'].rank(ascending=False, method='min').astype(int)
    print(result)
    return result

for year in years:
    TOPSIS(year).to_excel(f'../data/processed/{year}_TOPSIS.xlsx', index=False)