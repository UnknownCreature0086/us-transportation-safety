import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

years = [2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()
methods = ['TOPSIS', 'VIKOR', 'DOBI']

plt.rcParams['font.family'] = 'Times New Roman'

def obtain_performance(year, method):
    data = pd.read_excel(f'../data/processed/{year}_{method}.xlsx')
    performance = data[method]
    performance.index = data['State'].to_list()
    return performance

# normalized decision matrix
def normalize(matrix):
    column_sums = np.sum(matrix ** 2, axis=0)
    column_sums = column_sums ** 0.5
    matrix = matrix / column_sums
    return matrix

def normalize_horizontal(matrix):
    row_sums = np.sum(matrix ** 2, axis=1)
    row_sums = row_sums ** 0.5
    row_sums = row_sums.reshape(-1, 1)  # Reshape row_sums to be a column vector
    matrix = matrix / row_sums
    return matrix

# breakdown the data of each year
def breakdown(year):
    data = pd.read_excel(f'../data/processed/{year}_IDOCRIW_weighted_data.xlsx')[indicators].to_numpy()
    data = normalize_horizontal(data)
    data = pd.DataFrame(data)
    data.index = states
    data.columns = indicators
    return data


def plot_stacked_bar(year1, year2, method):
    # data preparation
    performance1 = obtain_performance(year1, method)
    performance2 = obtain_performance(year2, method)
    change = performance2 - performance1

    data1 = pd.read_excel(f'../data/processed/{year1}_IDOCRIW_weighted_data.xlsx')[indicators].to_numpy()
    data2 = pd.read_excel(f'../data/processed/{year2}_IDOCRIW_weighted_data.xlsx')[indicators].to_numpy()
    data = data2 - data1

    data = normalize_horizontal(data)
    data = pd.DataFrame(data)
    data.index = states
    data.columns = indicators

    for state in states:
        total_change = change[state]
        indicator_changes = data.loc[state]
        if indicator_changes.abs().sum() != 0:
            data.loc[state] = indicator_changes / indicator_changes.abs().sum() * total_change

    # separate indicators that increase and decrease
    positive_data = data.clip(lower=0)
    negative_data = data.clip(upper=0)

    min_value = negative_data.sum(axis=1).min() * 1.1
    max_value = positive_data.sum(axis=1).max() * 1.1

    colors = plt.cm.tab20(np.linspace(0, 1, len(indicators)))
    fig, ax = plt.subplots(figsize=(20, 30))

    left_positive = np.zeros(len(states))
    left_negative = np.zeros(len(states))

    # indicators that increase
    for i, indicator in enumerate(indicators):
        ax.barh(states, positive_data[indicator], left=left_positive, label=indicator, color=colors[i])
        left_positive += positive_data[indicator]

    # indicators that decrease
    for i, indicator in enumerate(indicators):
        ax.barh(states, negative_data[indicator], left=left_negative, color=colors[i])
        left_negative += negative_data[indicator]

    ax.axvline(0, color='black', linewidth=0.5, linestyle='-')
    ax.set_title(f'Year {year1}-{year2}', fontsize=44, pad=20)
    ax.set_xlabel(f'{method} Performance Change', fontsize=38)
    ax.set_ylabel('State', fontsize=38)
    ax.set_xlim(min_value, max_value)
    ax.set_ylim(-1, len(states))
    plt.xticks(fontsize=38)
    plt.yticks(fontsize=38)
    ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1), fontsize=36)
    plt.tight_layout()
    plt.savefig(f'../outputs/figures/deconstruct_change/{method}_change_{year1}to{year2}.jpg', bbox_inches='tight', dpi=400)
    plt.close()

# 示例调用
paired_years = [(year1, year2) for year1 in years for year2 in years if year2 > year1]
for method in methods:
    for (year1, year2) in paired_years:
        plot_stacked_bar(year1, year2, method)