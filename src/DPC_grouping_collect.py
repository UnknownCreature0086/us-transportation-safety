import pandas as pd

years = [2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()
methods = ['DPC', 'DPC_modified', 'DPC_remodified']

def collect(year):
    data = []
    for method in methods:
        grouping = pd.read_excel(f'../data/processed/{year}_{method}_groups.xlsx')['Group']
        grouping = grouping.rename(f'{method}')
        data.append(grouping)
    result = pd.concat(data, axis=1)
    result.insert(0, 'State', states)
    return result

for year in years:
    collect(year).to_excel(f'../data/processed/{year}_DPC.xlsx', index=False)
