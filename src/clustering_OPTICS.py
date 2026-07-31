import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.neighbors import NearestNeighbors

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()
plt.rcParams['font.family'] = 'Times New Roman'

# assign the noise points so that each country is in a certain cluster
def assign_noise_to_clusters(data, optics_labels):
    # locate the noise
    noise_mask = (optics_labels == -1)

    if not noise_mask.any():  # return the original label if the point is not a noise
        return optics_labels

    # find the index and tha label of the noise
    core_mask = ~noise_mask
    core_points = data[core_mask]
    core_labels = optics_labels[core_mask]

    # locate the nearest neighbour
    nbrs = NearestNeighbors(n_neighbors=1).fit(core_points)
    distances, indices = nbrs.kneighbors(data[noise_mask])

    # assign the noise to a cluster
    optics_labels[noise_mask] = core_labels[indices.flatten()]

    return optics_labels


# perform the DBSCAN process
def cluster_by_OPTICS(year):
    plt.clf()
    data = pd.read_excel(f'../data/processed/{year}_IDOCRIW_weighted_data.xlsx')[indicators]
    # preparation for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)
    # the OPTICS clustering
    optics = OPTICS(min_samples=2, xi=0.32, min_cluster_size=2)  # 调整参数
    optics_labels = optics.fit_predict(data)
    # assign the noise manually
    optics_labels = assign_noise_to_clusters(data, optics_labels)
    # visualization of the clustering
    plt.scatter((X_pca)[:, 0], X_pca[:, 1], c=optics_labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title(f'{year} OPTICS Clustering')
    plt.colorbar(label='Cluster Label')

    plt.tight_layout()

    sample_names = states
    labels = optics.labels_
    # Annotate each point with its sample name
    for i, name in enumerate(sample_names):
        plt.text(X_pca[i, 0], X_pca[i, 1], name, fontsize=8, ha='left')

    plt.savefig(f'../outputs/figures/OPTICS/{year}_OPTICS.jpg', bbox_inches='tight', dpi=600)

    score = pd.read_excel(f'../data/processed/{year}_DOBI.xlsx')

    # Group results by cluster
    unique_labels = set(labels)
    clusters = {label: [] for label in unique_labels}  # Initialize with all unique labels, including -1

    for i, name in enumerate(sample_names):
        cluster_id = labels[i]  # Get the cluster ID for this sample
        clusters[cluster_id].append(name)  # Add the sample to the corresponding cluster

    cluster_df = {i: [clusters[i]] for i in unique_labels}
    cluster_df = pd.DataFrame(cluster_df).transpose()
    cluster_df.columns = ['State']
    cluster_df['feature'] = cluster_df['State'].apply(lambda x: np.mean(score[score['State'].isin(x)]['Ranking']))
    cluster_df = cluster_df.sort_values(by='feature', ascending=True, ignore_index=True)
    cluster_df['real_label'] = list(range(1, len(unique_labels) + 1))

    sorted_clusters = dict(zip(cluster_df['real_label'].to_list(), cluster_df['State'].to_list()))

    # Alternatively, save all clusters into a single file with clear grouping
    optics_clusters_df = pd.concat([pd.DataFrame({'State': samples,
                                               'Group': [cluster_id] * len(samples),
                                               }) for cluster_id, samples in
                                 sorted_clusters.items()])  # sorted_clusters.items()
    print(optics_clusters_df)
    optics_clusters_df = pd.merge((pd.read_excel('../data/raw/2022data.xlsx'))['State'], optics_clusters_df, how='left', on='State')
    optics_clusters_df.to_excel(f'../data/processed/{year}_OPTICS_groups.xlsx', index=False)

for year in years:
    cluster_by_OPTICS(year)
