import pandas as pd

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
negative_indicators = ['X8']
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()

# read data
def read_data(year):
    df = pd.read_excel('../data/raw/US Transport Data_2010-2022.xlsx', sheet_name=f'{year}')
    df.columns = column_names
    df['State'] = states
    return df

for year in years:
    read_data(year).to_excel(f'../data/raw/{year}_raw.xlsx', index=False)