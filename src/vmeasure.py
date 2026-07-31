import pandas as pd
from sklearn.metrics import v_measure_score
from itertools import combinations
methods = ['DPC', 'KAT-DPC', 'kmeans']
years = [2016, 2019, 2022]


def get_grouping(year):
    results = pd.read_excel(f'../data/processed/{year}groups.xlsx')
    cluster_results = dict.fromkeys(methods)
    for method in methods:
        cluster_results[method] = results[method].to_list()
    return cluster_results

def compare_goruping(year):
    print(year)
    cluster_results = get_grouping(year)
    for (name1, labels1), (name2, labels2) in combinations(cluster_results.items(), 2):
        score = v_measure_score(labels1, labels2)
        print(f"{name1} vs {name2}: V-measure = {score:.3f}")

for year in years:
    compare_goruping(year)