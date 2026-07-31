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


def plot_stacked_bar(year, method):
    # data preparation
    performance = obtain_performance(year, method)
    data = breakdown(year)

    # scaling the data to match the performance score
    for state in states:
        total = data.loc[state].sum()
        if total != 0:  # to avoid zero division error
            data.loc[state] = data.loc[state] / total * performance[state]

    # color settings
    colors = plt.cm.tab20(np.linspace(0, 1, len(indicators)))

    fig, ax = plt.subplots(figsize=(18, 30))
    left = np.zeros(len(states))

    for i, indicator in enumerate(indicators):
        ax.barh(states, data[indicator], left=left, label=indicator, color=colors[i])
        left += data[indicator]

    ax.set_title(f'Year {year}', fontsize=48, pad=25)
    ax.set_xlabel(f'{method} Performance', fontsize=40)
    ax.set_ylabel('State', fontsize=40)
    ax.set_ylim(-1, len(states))
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=36)
    plt.tight_layout()
    plt.savefig(f'../outputs/figures/deconstruct_performance/{year}_{method}.jpg',bbox_inches='tight', dpi=400)
    plt.close()

for method in methods:
    for year in years:
        plot_stacked_bar(year, method)
