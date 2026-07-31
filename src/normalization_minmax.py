from sklearn.preprocessing import MinMaxScaler
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

# transform negative indicators
def transform_negative(year):
    data = read_data(year)
    data[negative_indicators] = 1 / data[negative_indicators]
    return data

# MinMax normalization
def MinMax_normalization(year):
    data = transform_negative(year)
    scaler = MinMaxScaler()
    data[indicators] = scaler.fit_transform(data[indicators])
    return data

for year in years:
    MinMax_normalization(year).to_excel(f'../data/processed/{year}_MinMax_normalized.xlsx', index=False)