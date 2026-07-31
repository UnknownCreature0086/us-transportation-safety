import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
negative_indicators = ['X8']
methods = ['DOBI', 'TOPSIS', 'VIKOR']
other_methods = ['TOPSIS', 'VIKOR']
plt.rcParams['font.family'] = 'Times New Roman'

def read_data(year):
    '''obtain data for drawing the line chart'''
    data = dict.fromkeys(methods)
    for method in methods:
        data[method] = pd.read_excel(f'../data/processed/{year}_{method}.xlsx')
    data['DOBI'] = data['DOBI'].sort_values(by='Ranking', ascending=True, ignore_index=True)
    sorted_states = data['DOBI']['State']
    dobi = data['DOBI'][['State', 'Ranking']]
    dobi = dobi.rename(columns={'Ranking': 'Ranking_DOBI'})
    for method in other_methods:
        data[method] = pd.merge(data[method], dobi, on='State', how='inner')
        data[method] = data[method].sort_values(by='Ranking_DOBI', ascending=True, ignore_index=True)
        del data[method]['Ranking_DOBI']
    return data, sorted_states

# obtain rankings
def obtain_rank(year):
    data = read_data(year)[0]
    rank = dict(zip(methods, [data[method]['Ranking'].to_numpy() for method in methods]))
    return rank

# calculate the coefficients
def calculate_correlation(year):
    rank = obtain_rank(year)
    ranks = np.vstack(list(rank.values()))
    corr_matrix = np.corrcoef(ranks)
    correlation = pd.DataFrame(corr_matrix)
    correlation.columns = methods
    correlation.insert(0, '', methods)
    for method in methods:
        correlation[method] = correlation[method].apply(lambda x: f'{x:.3f}')
    correlation.to_excel(f'../outputs/robustness/aggregating/{year}_correlation.xlsx', index=False)
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.axis('off')
    table = ax.table(
        cellText=correlation.values,  # Data for the table (from the DataFrame)
        colLabels=correlation.columns,  # Column headers (from the DataFrame)
        loc='center',  # Position the table at the center of the Axes
        cellLoc='center',  # Center-align the text in cells
        colColours=['#f3f3f3'] * len(correlation.columns),  # Light gray background for headers
        colWidths=[0.2] * len(correlation.columns)  # Adjust column widths
    )
    ax.set_title(f'Year{year}', fontweight='bold', fontfamily='Times New Roman', fontsize=8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_fontsize(6)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f3f3f3')
        elif col == 0:
            cell.set_fontsize(6)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f3f3f3')
        cell.set_text_props(fontfamily='Times New Roman', fontsize=6)
        cell.set_edgecolor('none')
    # Scale the table cells
    table.scale(1.2, 1.2)
    plt.savefig(f'../outputs/robustness/aggregating/{year}_correlation.jpg', bbox_inches='tight', dpi=600)


# draw the line chart
def line_chart(year):
    plt.rcParams.update({
        'font.size': 36,
        'axes.titlesize': 50,
        'axes.labelsize': 42,
        'xtick.labelsize': 42,
        'ytick.labelsize': 38,
        'legend.fontsize': 40,
    })
    rank = obtain_rank(year)
    sorted_states = read_data(year)[1]
    m = len(sorted_states)
    x = np.array(list(range(m)))
    plt.figure(figsize=(60,15))
    plt.xticks(x, sorted_states)
    plt.yticks(np.arange(1, m + 1), [str(i) if i % 2 != 0 else '' for i in range(1, m + 1)])
    plt.xlim(-1, m)
    plt.ylim(0, m + 1)
    plt.title(f'Year {year}', pad=20)
    plt.xlabel('State')
    plt.ylabel('Ranking')
    plt.grid(True)
    for state, method in rank.items():
        plt.plot(x, method, marker='o', markersize=10, linewidth=5, label=state)
        plt.legend(loc='lower right')
    plt.savefig(f'../outputs/robustness/aggregating/{year}_line.jpg', bbox_inches='tight', dpi=300)

# draw a table to compare the rankings
def draw_table(year):
    data = read_data(year)[0]
    for method in methods:
        data[method] = data[method].rename(columns={'Ranking': f'Ranking ({method})'})
        data[method][method] = data[method][method].apply(lambda x: f'{x:.3f}')
    result = reduce(lambda left, right: pd.merge(left, right, on='State', how='inner'), list(data.values()))
    result.to_excel(f'../outputs/robustness/aggregating/{year}_rankings.xlsx', index=False)
    with open(f'../outputs/robustness/aggregating/{year}_rankings.txt', 'w', encoding='utf-8') as file:
        file.write(result.to_latex(index=False, caption=f'Year{year}'))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    table = ax.table(
        cellText=result.values,  # Data for the table (from the DataFrame)
        colLabels=result.columns,  # Column headers (from the DataFrame)
        loc='center',  # Position the table at the center of the Axes
        cellLoc='center',  # Center-align the text in cells
        colColours=['#f3f3f3'] * len(result.columns),  # Light gray background for headers
        colWidths=[0.2] * len(result.columns)  # Adjust column widths
    )
    ax.set_title(f'Year {year}', fontweight='bold', fontfamily='Times New Roman', pad=250, fontsize=16)
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_fontsize(14)
            cell.set_text_props(weight='bold')
        elif col == 0:
            cell.set_fontsize(14)
            cell.set_text_props(weight='bold')
        cell.set_text_props(fontfamily='Times New Roman', fontsize=14)
        cell.set_edgecolor('none')
    # Scale the table cells
    table.scale(1.2, 1.2)
    plt.savefig(f'../outputs/robustness/aggregating/{year}_rankings.jpg', bbox_inches='tight', dpi=600)


for year in years:
    line_chart(year)
    draw_table(year)
    calculate_correlation(year)