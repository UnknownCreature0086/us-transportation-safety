import pandas as pd
import numpy as np

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()
column_names = ['State'] + indicators
negative_indicators = ['X8']
epsilon = 1e-8

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

# the first logarithmic normalization
def first_log(data):
    '''the first logarithmic normalization'''
    data[indicators] = data[indicators] + np.full_like(data[indicators], epsilon)
    data[indicators] = np.log(data[indicators])
    column_norm = np.sum(data[indicators], axis=0)
    data[indicators] = data[indicators] / column_norm
    return data

# the second logarithmic normalization
def second_log(data):
    '''the second logarithmic normalization'''
    data[indicators] = data[indicators] + np.full_like(data[indicators], epsilon)
    data[indicators] = np.log2(data[indicators])
    column_norm = np.sum(data[indicators], axis=0)
    data[indicators] = data[indicators] / column_norm
    return data

# calculate the Heron mean
def Heron_mean(year):
    '''calculate the Heron mean'''
    data = read_data(year)
    data1 = transform_negative(year)
    data2 = transform_negative(year)
    num1 = first_log(data1)[indicators]
    num2 = second_log(data2)[indicators]
    r = 0.2
    geometric_mean = (num1 * num2) ** 0.5
    arithmetic_mean = (num1 + num2) * 0.5
    data[indicators] = (1 - r) * geometric_mean + r * arithmetic_mean
    return data


def ARLON(year):
    '''the main ARLON process'''
    data = Heron_mean(year)
    weights = pd.read_excel(f'../data/processed/{year}_weights.xlsx')
    weights.columns = ['method'] + indicators
    weight = weights[weights['method'] == 'IDOCRIW'][indicators].to_numpy()
    data[indicators] = data[indicators] * weight
    index = np.sum(data[indicators], axis=1)
    result = pd.DataFrame({
        'State': states,
        'ARLON': index
    })
    result['Ranking'] = result['ARLON'].rank(ascending=False, method='min').astype(int)
    print(f'{year}\n', result)
    return result

for year in years:
    ARLON(year).to_excel(f'../data/processed/{year}_ARLON.xlsx', index=False)