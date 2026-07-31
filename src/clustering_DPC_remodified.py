import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

# Configuration and initialization
years = [2010, 2013, 2016, 2019, 2022]
indicators = [f'X{i}' for i in range(1, 9)]
states = (pd.read_excel('../data/raw/2022data.xlsx'))['State'].to_list()
plt.rcParams['font.family'] = 'Times New Roman'


def assign_noise_to_clusters(data, labels, density=None, max_noise_ratio=0.2):
    """Assign noise points to nearest non-noise clusters, considering density."""
    noise_mask = (labels == -1)
    if not noise_mask.any() or np.sum(noise_mask) / len(data) > max_noise_ratio:
        return labels

    if density is not None:
        noise_mask = noise_mask & (density > np.mean(density))

    core_points = data[~noise_mask]
    core_labels = labels[~noise_mask]
    nbrs = NearestNeighbors(n_neighbors=1).fit(core_points)
    _, indices = nbrs.kneighbors(data[noise_mask])
    labels[noise_mask] = core_labels[indices.flatten()]
    return labels


def enforce_cluster_count(gamma, target_range=(8, 9)):
    """Ensure we get clusters within target range."""
    gamma_sorted = -np.sort(-gamma)
    kneedle = KneeLocator(range(len(gamma_sorted)), gamma_sorted,
                          curve='convex', direction='decreasing')

    if kneedle.elbow is not None:
        n_clusters = kneedle.elbow + 1
        if n_clusters < target_range[0]:
            return target_range[0]
        elif n_clusters > target_range[1]:
            return target_range[1]
        return n_clusters
    return target_range[0]  # Default to minimum target


def cluster_by_DPC(year):
    """Main DPC clustering function with removed small cluster merging."""
    plt.clf()
    data = pd.read_excel(f'../data/processed/{year}_IDOCRIW_weighted_data.xlsx')[indicators]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)

    # 1. Density calculation
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    knn_distances, _ = nbrs.kneighbors(data)
    density = 1 / (np.mean(knn_distances, axis=1) + 1e-10)

    # 2. Delta calculation
    distances = squareform(pdist(data))
    delta = np.zeros_like(density)
    for i in range(len(data)):
        higher_density = density > density[i]
        if np.any(higher_density):
            delta[i] = np.min(distances[i, higher_density])
        else:
            delta[i] = np.max(distances[i])

    # 3. Cluster count selection
    gamma = density * delta
    n_clusters = enforce_cluster_count(gamma)

    # 4. Center selection
    centers = np.argsort(-gamma)[:n_clusters]
    kmeans = KMeans(n_clusters=n_clusters, init=data.iloc[centers],
                    n_init=1, random_state=42)
    dpc_labels = kmeans.fit_predict(data)

    # 5. Only assign noise (no cluster merging)
    dpc_labels = assign_noise_to_clusters(data, dpc_labels, density)

    # Evaluation
    if len(np.unique(dpc_labels)) > 1:
        silhouette_avg = silhouette_score(data, dpc_labels)
        print(f"{year} Silhouette Score: {silhouette_avg:.3f} (n_clusters={n_clusters})")
        print(f"Actual unique clusters: {len(np.unique(dpc_labels))}")

    # Visualization
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dpc_labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title(f'{year} DPC Clustering')
    plt.colorbar(label='Cluster Label')
    for i, name in enumerate(states):
        plt.text(X_pca[i, 0], X_pca[i, 1], name, fontsize=8, ha='left')
    plt.savefig(f'../outputs/figures/DPC/{year}_DPC_remodified.jpg', bbox_inches='tight', dpi=400)
    plt.close()

    # 10. Save results
    score = pd.read_excel(f'../data/processed/{year}_DOBI.xlsx')
    unique_labels = set(dpc_labels)
    clusters = {label: [] for label in unique_labels}

    for i, name in enumerate(states):
        clusters[dpc_labels[i]].append(name)

    # Sort clusters by average ranking
    cluster_df = pd.DataFrame({
        'State': [clusters[label] for label in unique_labels],
        'Label': list(unique_labels)
    })
    cluster_df['AvgRank'] = cluster_df['State'].apply(
        lambda x: np.mean(score[score['State'].isin(x)]['Ranking']))
    cluster_df = cluster_df.sort_values('AvgRank').reset_index(drop=True)
    cluster_df['Group'] = range(1, len(cluster_df) + 1)

    # Expand to one row per state
    result_df = pd.concat([
        pd.DataFrame({'State': states, 'Group': group})
        for states, group in zip(cluster_df['State'], cluster_df['Group'])
    ])
    print(result_df)
    result_df = pd.merge((pd.read_excel('../data/raw/2022data.xlsx'))['State'], result_df, how='left', on='State')
    result_df.to_excel(f'../data/processed/{year}_DPC_remodified_groups.xlsx', index=False)



# Execute clustering
for year in years:
    cluster_by_DPC(year)