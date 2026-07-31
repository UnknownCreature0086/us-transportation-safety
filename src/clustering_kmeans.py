import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()
plt.rcParams['font.family'] = 'Times New Roman'


def cluster_by_kmeans(year):
    plt.clf()
    data = pd.read_excel(f'../data/processed/{year}_IDOCRIW_weighted_data.xlsx')[indicators]

    # Elbow method to choose K (optional - you can keep this or set fixed K=5)
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.show()
    plt.close()

    # Perform K-means clustering with K=5
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(data)
    labels = kmeans.labels_

    # Visualization (using PCA for 2D projection)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis')
    plt.title(f'{year} K-means Clustering')
    plt.colorbar(label='Cluster Label')

    # Annotate each point with state name
    for i, name in enumerate(states):
        plt.text(data_pca[i, 0], data_pca[i, 1], name, fontsize=8, ha='left')

    plt.savefig(f'../outputs/figures/kmeans/{year}_kmeans.jpg', bbox_inches='tight', dpi=600)
    plt.close()

    # Load ranking data
    score = pd.read_excel(f'../data/processed/{year}_DOBI.xlsx')

    # Group results by cluster (using DBSCAN-style organization)
    unique_labels = set(labels)
    clusters = {label: [] for label in unique_labels}

    for i, name in enumerate(states):
        cluster_id = labels[i]
        clusters[cluster_id].append(name)

    # Create DataFrame with cluster information
    cluster_df = {i: [clusters[i]] for i in unique_labels}
    cluster_df = pd.DataFrame(cluster_df).transpose()
    cluster_df.columns = ['State']
    cluster_df['feature'] = cluster_df['State'].apply(
        lambda x: np.mean(score[score['State'].isin(x)]['Ranking']))
    cluster_df = cluster_df.sort_values(by='feature', ascending=True, ignore_index=True)
    cluster_df['real_label'] = list(range(1, len(unique_labels) + 1))

    # Create final output DataFrame (matching DBSCAN format)
    sorted_clusters = dict(zip(cluster_df['real_label'].to_list(), cluster_df['State'].to_list()))
    kmeans_clusters_df = pd.concat([
        pd.DataFrame({'State': states, 'Group': [cluster_id] * len(states)})
        for cluster_id, states in sorted_clusters.items()
    ])
    print(kmeans_clusters_df)

    kmeans_clusters_df = pd.merge((pd.read_excel('../data/raw/2022data.xlsx'))['State'], kmeans_clusters_df, how='left', on='State')

    # Save results
    kmeans_clusters_df.to_excel(f'../data/processed/{year}_kmeans_groups.xlsx', index=False)

    return kmeans_clusters_df


for year in years:
    cluster_by_kmeans(year)