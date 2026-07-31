import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()
methods = ['DPC', 'DPC_modified', 'DPC_remodified']
gathered_data = {method: {} for method in methods}


# color settings
# 合并 tab20 和 tab20b（共40种）
colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
colors_tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))
colors_40 = np.vstack((colors_tab20, colors_tab20b))
# 补充10种其他高对比色（来自Set3/Paired）
colors_extra = plt.cm.Set3(np.linspace(0, 1, 12))[:10]  # 取前10种
colors_50 = np.vstack((colors_40, colors_extra))
# 创建Colormap对象
cmap_50 = ListedColormap(colors_50, name='HighContrast50')


# for forming the radar chart lines
def close_data(data):
    return np.concatenate((data, [data[0]]))

def obtain_data(year, method, group):
    group_data = gathered_data[method][year][group]
    values = [close_data(np.array(group_data.loc[country].to_list())) for country in group_data.index]
    max_value = max([max(v) for v in values]) * 1.1
    min_value = min([min(v) for v in values]) * 0.9
    return min_value, max_value, values

# initializing
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(polar=True))


# find the grouped countries
def obtain_groups(year, method):
    gathered_data[method][year] = {}
    data = pd.read_excel(f'../data/processed/{year}_IDOCRIW_weighted.xlsx')[indicators]
    data.index = states
    grouping_result = pd.read_excel(f'../data/processed/{year}_{method}_groups.xlsx')
    groups = grouping_result['Group'].unique()
    for group in groups:
        country_within_group = grouping_result[grouping_result['Group'] == group]['State'].to_list()
        group_data = data[data.index.isin(country_within_group)]
        gathered_data[method][year][f'group {group}'] = group_data
    return gathered_data

# create a pipeline for drawing radar chart for each set of data
def radar(year, method, group): # the 'group' here is the name of a particular group
    ax.clear()
    whole_data = gathered_data[method]  # to be continued...
    states_within_group = whole_data[year][group].index
    # calculate the angles
    labels = np.array(indicators)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the angles
    # data input
    min_value, max_value, values = obtain_data(year, method, group)
    for i in range(len(values)):
        ax.plot(angles, values[i], color=cmap_50(i / 49), linewidth=4, label=states_within_group[i], marker='o', markersize=10)
    # set labels
    ax.tick_params(axis='y', labelsize=32)  # determine fontsize of y labels
    ax.set_ylim(min_value, max_value)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=40, fontweight='bold')
    ax.tick_params(axis='both', pad=25)
    # 优化图例显示 - 自动分列
    n_states = len(states_within_group)
    if n_states > 10:
        n_cols = 2
        a, b = (1.6, 1.15)
    else:
        n_cols = 1
        a, b = (1.32, 1.05)
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(a, b),  # 稍微调整位置以适应多列
        fontsize=40,
        ncol=n_cols,  # 设置列数
        framealpha=0.9  # 添加半透明背景
    )
    ax.set_title(f'{group}'.capitalize(), fontweight='bold', fontfamily='Times New Roman', pad=90, fontsize=48)
    plt.savefig(f'../outputs/figures/decompose_DPC_group/{method}_{year}_{group}.jpg', bbox_inches='tight', dpi=400)
    return

# the final radar charts
def radar_groups(year, method):
    whole_data = gathered_data[method] # to be continued...
    groups = list(whole_data[year].keys())
    for group in groups:
        radar(year, method, group)
    return

def gathering():
    for year in years:
        for method in methods:
            obtain_groups(year, method)
    return
gathering()

for method in methods:
    for year in years:
        radar_groups(year, method)

