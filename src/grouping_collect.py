import pandas as pd
from functools import reduce

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()
methods = ['DPC', 'kmeans', 'DBSCAN', 'HDBSCAN', 'OPTICS', 'RSR']

def collect(year):
    data = []
    for method in methods:
        grouping = pd.read_excel(f'../data/processed/{year}_{method}_groups.xlsx')['Group']
        grouping = grouping.rename(f'{method}')
        data.append(grouping)
    result = pd.concat(data, axis=1)
    result.insert(0, 'State', states)
    print(result.head())
    return result

for year in years:
    collect(year).to_excel(f'../data/processed/{year}_groups.xlsx', index=False)
