import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

# zscore normalization
def zscore_normalization(year):
    data = transform_negative(year)
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(data[indicators])
    a, b = 0, 1
    z_scores = (z_scores - np.min(z_scores)) / (np.max(z_scores) - np.min(z_scores)) * (b - a) + a
    data[indicators] = z_scores
    return data


for year in years:
    zscore_normalization(year).to_excel(f'../data/processed/{year}_zscore_normalized.xlsx', index=False)