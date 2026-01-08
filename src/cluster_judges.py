from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import kruskal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

from .feature_stats import generate_stat_table

def bic_kmeans(kmeans, X):
    # Derived from: Pelleg, Dan, and Andrew W. Moore. "X-means: Extending k-means
    # with efficient estimation of the number of clusters." Icml. Vol. 1. 2000.
    # and https://github.com/bobhancock/goxmeans/blob/master/doc/BIC_notes.pdf

    R = X.shape[0]
    labels = kmeans.labels_
    Rn = np.bincount(labels)
    M = X.shape[1]
    k = kmeans.n_clusters

    variance = kmeans.inertia_ / ((R - k) * M)
    pj = (k - 1) + (M * k) + 1

    log_likelihood = np.sum(
            ((Rn*np.log(Rn)) - (Rn*np.log(R))) \
        - ((Rn*M)/2)*np.log(2*np.pi*variance) \
        - ((M/2)*(Rn-1)))

    bic = log_likelihood - (pj / 2.0) * np.log(R)
    return bic

def liking_correlation_scores(data):
    model = GradientBoostingRegressor(random_state=42)
    stats_judges = pd.DataFrame()
    for judge in data['Judge'].unique():
        X = data[data['Judge']==judge].drop(columns=['Judge', 'Liking'])
        y = data[data['Judge']==judge]['Liking']
        stats = generate_stat_table(X, y, True, True, model, include_rfe=True)
        stats['Judge'] = judge
        stats_judges = pd.concat([stats_judges, stats], axis=0)
        stats_judges.to_csv('data/processed/stats_for_each_judge.csv')

    return stats_judges
    
def cluster_judges_kmeans(data):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Get explained variance ratio
    explained_variance_ratio = PCA().fit(scaled_data).explained_variance_ratio_

    # Use KneeLocator to find the elbow
    # x values are the number of components (1 to the total number of components)
    x_values = range(1, len(explained_variance_ratio) + 1)
    kl = KneeLocator(x_values, explained_variance_ratio, curve='convex', direction='decreasing')

    # Get the elbow point
    pca_elbow_point = kl.elbow

    sns.set_style('white')
    # Generate the scree plot with the elbow marked
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=x_values, y=explained_variance_ratio, marker='o')
    plt.title('PCA Elbow Plot', fontsize=22)
    plt.xlabel('Number of Components', fontsize=20)
    plt.ylabel('Explained Variance Ratio', fontsize=20)

    if pca_elbow_point is not None:
        plt.axvline(pca_elbow_point, color='black', linestyle='--', label=f'Elbow at Component {pca_elbow_point}')
        plt.legend(fontsize=18)
        plt.tick_params(axis='both', labelsize=16) 
        plt.xticks(x_values)
        pca_n_dims = PCA(n_components=pca_elbow_point)
        pca_n_dims_result = pca_n_dims.fit_transform(scaled_data)
    else:
        pca_n_dims_result = scaled_data


    plt.savefig("figures/supplementary/PCA_elbow.svg")
    plt.close()

    k_range = range(2, 21)
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans_result = kmeans.fit(pca_n_dims_result)
        scores.append({"Distortion\n(Lower = better)":kmeans.inertia_,
                       "Silhouette\n(Higher = better)":silhouette_score(pca_n_dims_result, kmeans.labels_),
                       "Calinski-Harabasz\n(Higher = better)":calinski_harabasz_score(pca_n_dims_result, kmeans.labels_),
                       "BIC\n(Higher = better)":bic_kmeans(kmeans, pca_n_dims_result)})
    scores = pd.DataFrame(scores)
    for i, col, in enumerate(scores.columns):
        coord = np.unravel_index(i, (2,2))
        ax = axes[coord]
        sns.lineplot(data=scores, x=k_range, y=col, marker='D', ax=ax)
        if "Distortion" in col:
            kl = KneeLocator(k_range, scores[col], curve='convex', direction='decreasing')
        else:
            kl = KneeLocator(k_range, scores[col], curve='concave', direction='increasing')

        elbow_point = kl.elbow
        if elbow_point is not None:
            ax.axvline(elbow_point, color='black', linestyle='--', label=f'Elbow at k={elbow_point}')
            ax.legend(fontsize=18)
            pca_n_dims = PCA(n_components=elbow_point)
            pca_n_dims_result = pca_n_dims.fit_transform(scaled_data)

        ax.set_xlabel('k', fontsize=18)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)
        ax.set_xticks(range(2, 21, 2))
        ax.tick_params(axis='both', labelsize=16) 
    plt.tight_layout()
    plt.savefig("figures/supplementary/kmeans_k.svg")
    plt.close()



    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_result = kmeans.fit_predict(pca_n_dims_result)

    pca_2_dims = PCA(n_components=2)
    pca_2_dims_result = pca_2_dims.fit_transform(scaled_data)
    pca_df = pd.DataFrame(pca_2_dims_result, columns=['PC1', 'PC2'])

    sns.set_style("whitegrid", {'axes.grid' : False})
    ax = sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue=kmeans_result, palette=sns.color_palette('pastel')[2:])
    plt.legend(loc='upper right', labels=['Cluster 1', 'Cluster 2'])
    plt.title('PCA with KMeans Clustering Results', fontsize=20)
    plt.xlabel('Principal Component 1', fontsize=16)
    plt.ylabel('Principal Component 2', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)

    plt.savefig("figures/main/PCA_elbow.svg")
    plt.close()

    return kmeans_result

def cluster_stats(data, clusters):
    feature_scores = {}
    clustered = data.copy()
    clustered['judge_cluster'] = clusters
    for feature in clustered.columns.difference(['judge_cluster']):
        groups = [clustered[clustered['judge_cluster'] == c][feature] for c in clustered['judge_cluster'].unique()]
        stat, p = kruskal(*groups)
        feature_scores[feature] = p

    feature_means = clustered.groupby('judge_cluster').mean() 
    feature_means.loc['pvalue'] = feature_scores.values()
    feature_means = feature_means.T.sort_values(by='pvalue').T
    feature_means.to_csv('data/processed/cluster_feature_means.csv')


def main():
    cof = pd.read_csv('data/cotter_dataset.csv')
    cof_jar = cof[['Flavor.intensity', 'Acidity', 'Mouthfeel']]
    cof_jar_trans = abs(cof_jar-3)
    cof_jar_trans.columns = ['Flavor intensity (adj)', 'Acidity (adj)', 'Mouthfeel (adj)']
    cof_cata = cof.loc[:, 'Tea.floral':]
    cof_cata.drop(columns='Purchase.intent', inplace=True)
    cof_sensory = pd.concat([cof_cata, cof_jar_trans, cof[['Judge', 'Liking']]], axis=1)
    rerun = False
    if rerun:
        stats_judges = liking_correlation_scores(cof_sensory)
        stats_judges.to_csv('data/processed/stats_for_each_judge.csv')
    else:
        stats_judges = pd.read_csv('data/processed/stats_for_each_judge.csv', index_col=0)

    stats_judges.loc[stats_judges['SRC pval'].isna(), 'SRC'] = 0
    judge_liking_corr = stats_judges.pivot_table(index='Judge', columns=stats_judges.index, values='SRC')
    kmeans_result = cluster_judges_kmeans(judge_liking_corr)
    kmeans_result += 1
    cluster_stats(judge_liking_corr, kmeans_result)
    # cluster_stats(judge_liking_corr, cof['Cluster'])
    judge_liking_corr['judge_cluster'] = kmeans_result

    cof_clustered = pd.merge(cof, judge_liking_corr['judge_cluster'], left_on='Judge', right_index=True)
    cof_clustered.to_csv('data/processed/cof_with_judge_clusters.csv', index=False)

if __name__ == "__main__":
    main()