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


def merge_small_clusters(data, labels, min_size=1):
    """Merge clusters smaller than min_size into nearest larger clusters."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    small_clusters = unique_labels[counts < min_size]

    for sc in small_clusters:
        mask = (labels == sc)
        if np.sum(mask) == 0:
            continue
        nbrs = NearestNeighbors(n_neighbors=1).fit(data[~mask])
        _, indices = nbrs.kneighbors(data[mask])
        nearest_labels = labels[~mask][indices.flatten()]
        labels[mask] = nearest_labels
    return labels


def enforce_cluster_count(gamma, target_range=(5, 6)):
    """
    Modified version to better hit target range by:
    1. Using a more flexible knee detection
    2. Adding a fallback to select top candidates when knee detection fails
    """
    gamma_sorted = -np.sort(-gamma)

    # Try knee detection first
    kneedle = KneeLocator(range(len(gamma_sorted)), gamma_sorted,
                          curve='convex', direction='decreasing')

    # If we get a clear knee point
    if kneedle.elbow is not None:
        n_clusters = kneedle.elbow + 1
        if n_clusters < target_range[0]:
            # If too few, take top candidates from target range
            return min(target_range[1], len(gamma_sorted))
        elif n_clusters > target_range[1]:
            # If too many, take midpoint
            return target_range[1]
        else:
            return n_clusters
    else:
        # Fallback: Take midpoint of target range
        return min(target_range[1], len(gamma_sorted) // 2 + 1)


def cluster_by_DPC(year):
    """Modified clustering function with enhanced center selection"""
    plt.clf()

    # 1. Load and prepare data
    data = pd.read_excel(f'../data/processed/{year}_IDOCRIW_weighted_data.xlsx')[indicators]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)

    # 2. Improved density calculation
    k = 2  # Number of neighbors for density estimation
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    knn_distances, _ = nbrs.kneighbors(data)
    # Adjust the density calculation to be more discriminative
    density = 1 / (np.percentile(knn_distances, 25, axis=1) + 1e-10)  # Use 25th percentile distance

    # 3. Calculate delta
    distances = squareform(pdist(data))
    delta = np.zeros_like(density)
    for i in range(len(data)):
        higher_density = density > density[i]
        if np.any(higher_density):
            delta[i] = np.min(distances[i, higher_density])
        else:
            delta[i] = np.max(distances[i])

    # 4. Enforce 8-9 clusters
    gamma = density * delta
    n_clusters = enforce_cluster_count(gamma, target_range=(8, 9))

    # Enhanced center selection
    centers = []
    gamma_sorted_idx = np.argsort(-gamma)

    # First select the top 2 most obvious centers
    centers.extend(gamma_sorted_idx[:2])

    # Then select remaining centers spaced apart
    min_distance = np.median(distances) * 0.5  # Minimum distance between centers
    for i in range(2, len(gamma_sorted_idx)):
        if len(centers) >= n_clusters:
            break
        candidate = gamma_sorted_idx[i]
        # Check distance to existing centers
        if all(distances[candidate, c] > min_distance for c in centers):
            centers.append(candidate)

    # If we didn't get enough centers, add the next best candidates
    while len(centers) < n_clusters and len(centers) < len(gamma_sorted_idx):
        next_candidate = gamma_sorted_idx[len(centers)]
        centers.append(next_candidate)

    # Modified K-Means initialization
    if len(centers) >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, init=data.iloc[centers[:n_clusters]],
                        n_init=1, random_state=42)
    else:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                        n_init=10, random_state=42)

    dpc_labels = kmeans.fit_predict(data)

    # 7. Post-processing
    dpc_labels = merge_small_clusters(data, dpc_labels, min_size=1)  # Allow smaller min_size since we enforce count
    dpc_labels = assign_noise_to_clusters(data, dpc_labels, density)

    # 8. Evaluation
    if len(np.unique(dpc_labels)) > 1:
        silhouette_avg = silhouette_score(data, dpc_labels)
        print(f"{year} Silhouette Score: {silhouette_avg:.3f}")

    # 9. Visualization
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dpc_labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title(f'{year} DPC Clustering')
    plt.colorbar(label='Cluster Label')
    for i, name in enumerate(states):
        plt.text(X_pca[i, 0], X_pca[i, 1], name, fontsize=8, ha='left')
    plt.savefig(f'../outputs/figures/DPC/{year}_DPC_modified.jpg', bbox_inches='tight', dpi=400)
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
    result_df.to_excel(f'../data/processed/{year}_DPC_modified_groups.xlsx', index=False)

# Execute clustering for all years
for year in years:
    cluster_by_DPC(year)