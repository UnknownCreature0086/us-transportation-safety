import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
column_names = ['State'] + indicators
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()
plt.rcParams['font.family'] = 'Times New Roman'

# assign the noise points so that each country is in a certain cluster
def assign_noise_to_clusters(data, dpc_labels):
    # locate the noise
    noise_mask = (dpc_labels == -1)

    if not noise_mask.any():  # return the original label if the point is not a noise
        return dpc_labels

    # find the index and tha label of the noise
    core_mask = ~noise_mask
    core_points = data[core_mask]
    core_labels = dpc_labels[core_mask]

    # locate the nearest neighbour
    nbrs = NearestNeighbors(n_neighbors=1).fit(core_points)
    distances, indices = nbrs.kneighbors(data[noise_mask])

    # assign the noise to a cluster
    dpc_labels[noise_mask] = core_labels[indices.flatten()]

    return dpc_labels

# perform the DPC process
def cluster_by_DPC(year):
    plt.clf()
    data = pd.read_excel(f'../data/processed/{year}_IDOCRIW_weighted_data.xlsx')[indicators]
    # Preparation for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)

    # Calculate the density and distances
    distances = squareform(pdist(data))

    # Set cutoff distance (dc) with a minimum value to avoid division by zero
    dc = max(np.percentile(distances, 15), 1e-5)  # Ensure dc is not too small

    # Calculate density with a small epsilon to avoid division by zero
    epsilon = 1e-10
    density = np.sum(np.exp(-((distances + epsilon) / dc) ** 2), axis=1)

    # Calculate delta (minimum distance to a point with higher density)
    delta = np.zeros_like(density)
    for i in range(len(data)):
        higher_density_mask = density > density[i]
        if np.any(higher_density_mask):
            delta[i] = np.min(distances[i, higher_density_mask])
        else:
            delta[i] = np.max(distances[i])  # Assign the maximum distance if no higher density points

    # Select the centroids of clusters
    n_clusters = 8  # Assume 5 clusters
    centers = np.argsort(-density * delta)[:n_clusters]

    # Clustering based on k-means
    kmeans = KMeans(n_clusters=n_clusters, init=data.iloc[centers])
    dpc_labels = kmeans.fit_predict(data)

    # Assign the noise manually
    dpc_labels = assign_noise_to_clusters(data, dpc_labels)

    # Visualization of the clustering
    plt.scatter((X_pca)[:, 0], X_pca[:, 1], c=dpc_labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title(f'{year} DPC Clustering')
    plt.colorbar(label='Cluster Label')
    plt.tight_layout()

    # Annotate each point with its sample name
    sample_names = states
    for i, name in enumerate(sample_names):
        plt.text(X_pca[i, 0], X_pca[i, 1], name, fontsize=8, ha='left')

    plt.savefig(f'../outputs/figures/DPC/{year}_DPC.jpg', bbox_inches='tight', dpi=400)

    # Group results by cluster
    score = pd.read_excel(f'../data/processed/{year}_DOBI.xlsx')
    unique_labels = set(dpc_labels)
    clusters = {label: [] for label in unique_labels}

    for i, name in enumerate(sample_names):
        cluster_id = dpc_labels[i]
        clusters[cluster_id].append(name)

    cluster_df = {i: [clusters[i]] for i in unique_labels}
    cluster_df = pd.DataFrame(cluster_df).transpose()
    cluster_df.columns = ['State']
    cluster_df['feature'] = cluster_df['State'].apply(lambda x: np.mean(score[score['State'].isin(x)]['Ranking']))
    cluster_df = cluster_df.sort_values(by='feature', ascending=True, ignore_index=True)
    cluster_df['real_label'] = list(range(1, len(unique_labels) + 1))

    sorted_clusters = dict(zip(cluster_df['real_label'].to_list(), cluster_df['State'].to_list()))

    # Save all clusters into a single file
    dpc_clusters_df = pd.concat([pd.DataFrame({'State': samples, 'Group': [cluster_id] * len(samples)}) for cluster_id, samples in sorted_clusters.items()])
    print(dpc_clusters_df)
    dpc_clusters_df = pd.merge((pd.read_excel('../data/raw/2022data.xlsx'))['State'], dpc_clusters_df, how='left', on='State')
    dpc_clusters_df.to_excel(f'../data/processed/{year}_DPC_groups.xlsx', index=False)

for year in years:
    cluster_by_DPC(year)

