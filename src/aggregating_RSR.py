import pandas as pd
import numpy as np

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
    RSRw = (np.sum(data, axis=1)) / m
    return RSRw

def RSR(year):
    data = pd.read_excel(f'../data/processed/{year}_IDOCRIW_weighted_data.xlsx')[indicators].to_numpy()
    index = obtain_RSRw(obtain_rank(data))
    result = pd.DataFrame({
        'State': states,
        'RSR': index
    })
    result['Ranking'] = result['RSR'].rank(ascending=False, method='min').astype(int)
    print(f'{year}:\n', result)
    return result

for year in years:
    RSR(year).to_excel(f'../data/processed/{year}_RSR.xlsx', index=False)