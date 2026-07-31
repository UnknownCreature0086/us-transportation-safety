import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

years = [2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()

# for forming the radar chart lines
def close_data(data):
    return np.concatenate((data, [data[0]]))

def obtain_real_data(indicator):
    values = [close_data(np.array(pd.read_excel(f'../data/processed/{year}_raw.xlsx')[indicator].to_list())) for year in years]
    return values

def obtain_normalized_data(indicator):
    normalized_values = [close_data(np.array(pd.read_excel(f'../data/processed/{year}_IDOCRIW_weighted_data.xlsx')[indicator].to_list())) for year in years]
    return normalized_values

# calculate the angles
labels = np.array(states)
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the angles

# initializing
plt.rcParams['font.family'] = 'Times New Roman'
colormap = plt.cm.tab20
colors = [colormap(i * 2) for i in np.linspace(0, 1, 20)]

# draw the radar chart
def radar_real(indicator):
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(polar=True))
    values = obtain_real_data(indicator)
    max_value = max([max(v) for v in values])
    min_value = min([min(v) for v in values])
    for i in range(len(values)):
        ax.plot(angles, values[i], color=colors[i], linewidth=5, label=str(years[i]), marker='o', markersize=10)
    # set labels
    ax.tick_params(axis='y', labelsize=32)  # determine fontsize of y labels
    ax.set_ylim(min_value, max_value)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=40, fontweight='bold')
    ax.tick_params(axis='both', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=40)
    ax.set_title(indicator, fontweight='bold', fontfamily='Times New Roman', pad=110, fontsize=48)
    return fig

def radar_normalized(indicator):
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(polar=True))
    values = obtain_normalized_data(indicator)
    max_value = max([max(v) for v in values])
    min_value = min([min(v) for v in values])
    for i in range(len(values)):
        ax.plot(angles, values[i], color=colors[i], linewidth=2, label=str(years[i]), marker='o')
    # set labels
    ax.set_ylim(min_value, max_value)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=36, fontweight='bold')
    ax.tick_params(axis='both', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=36)
    ax.set_title(f'{indicator}(normalized)', fontweight='bold', fontfamily='Times New Roman', pad=80, fontsize=36)
    return fig

'''# 存储所有图形对象
figures = []

for indicator in indicators:
    figures.append(radar_normalized(indicator))
    figures.append(radar_real(indicator))

# 等待所有图形显示
plt.show()

# 关闭所有图形
for fig in figures:
    plt.close(fig)'''

for indicator in indicators:
    radar_real(indicator).savefig(f'../outputs/figures/decompose_indicator/{indicator}.jpg', bbox_inches='tight', dpi=500)
    radar_normalized(indicator).savefig(f'../outputs/figures/decompose_indicator/{indicator}_normalized.jpg', bbox_inches='tight', dpi=500)