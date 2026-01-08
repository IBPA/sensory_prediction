import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest
from scipy.stats import spearmanr
from statannotations.Annotator import Annotator
from matplotlib_venn import venn2


from .constants import *

def plot_drivers_of_liking(cof_features, cof_features_brew, cof_features_judge,
                           cof_stats, cof_stats_brew, cof_stats_judge, output_path):
    sns.set_style("whitegrid", {'axes.grid' : False})
    _, axes = plt.subplots(nrows=3, figsize=(5, 12))
    # plt.subplots_adjust(hspace=0.5)
    palette = sns.color_palette("pastel")

    X = cof_features[cof_stats.iloc[0].name]
    y = cof_features['liking']
    ax = sns.boxplot(x=X, y=y, ax=axes[0], color=palette[0])

    X = cof_features_brew[cof_stats_brew.iloc[0].name]
    y = cof_features_brew['liking']
    sns.scatterplot(x=X, y=y, ax=axes[1], alpha=0.8, color=palette[1])

    X = cof_features_judge[cof_stats_judge.iloc[0].name]
    y = cof_features_judge['liking']
    sns.scatterplot(x=X, y=y, ax=axes[2], alpha=0.8, color=palette[2])

    x_labels = [cof_stats.iloc[0].name,
                cof_stats_brew.iloc[0].name,
                cof_stats_judge.iloc[0].name]
    for ax, x_label in zip(axes, x_labels):
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel('Liking', fontsize=22)
        ax.tick_params(axis='both', labelsize=18) 

    plt.tight_layout()
    # plt.tight_layout(h_pad=5.0)
    plt.savefig(output_path)
    plt.close()

def plot_jar_features(jar_data, y, output_path):
    sns.set_style("whitegrid", {'axes.grid' : False})
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 4), sharey=True)
    for i, jar_attribute in enumerate(jar_data.columns):
        coord = (i,)
        sns.boxplot(x=jar_data[jar_attribute], y=y, ax=axes[coord], color=sns.palettes.color_palette('pastel')[0])
        axes[coord].set_xlabel(jar_attribute.replace('.', ' '), fontsize=14)
        axes[0].set_ylabel('Liking', fontsize=14)
        pairs=[(1,5),(2,4)]
        annotator = Annotator(axes[coord], pairs, x=jar_data[jar_attribute], y=y)
        annotator.configure(test='Kruskal', text_format='star', loc='outside', verbose=0)
        annotator.apply_and_annotate()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_liking_corr(cof_stats, cof_stats_brew, cof_stats_judge, output_path):
    sns.set_style("whitegrid", {'axes.grid' : False})
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3), sharey=True)
    plt.subplots_adjust(wspace=0.05)
    features_selected = [14, 8, 8]
    title = ['No aggregation', 'Brew score aggregation', 'Judge score aggregation']
    for i, stat_table in enumerate([cof_stats, cof_stats_brew, cof_stats_judge]):
        sns.set_style("whitegrid")
        data = stat_table.sort_values(by='SRC', key=abs, ascending=False)
        data = data[data['SRC pval'] < 0.05]['SRC'][:features_selected[i]]
        sns.barplot(x=data.index, y=data.values, hue=data.values, palette='coolwarm', legend=False, ax=axes[i])
        axes[i].set_xticks(range(len(data.index)))
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90, ha='center')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Spearman Rank Correlation', fontsize=10)
        axes[i].set_title(title[i])
    plt.savefig(output_path)
    plt.close()

def plot_cata_corrmap(cata_data, output_path):
    cata_data.rename(columns={'liking':'Liking'}, inplace=True)
    g = sns.clustermap(cata_data.corr(method='spearman'),
            # annot=True,
            cmap='coolwarm',
            vmin=-1, vmax=1,
            # annot_kws={"size": 6},
            cbar_kws={"label": "Spearman correlation"},
            cbar_pos=(0.8, 0.28, 0.03, 0.4),
            # cbar_pos=(0.3, 0.1, 0.4, 0.03),
            linewidths=2, 
            linecolor='white')
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=18)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_xticklabels(), fontsize=18)
    plt.savefig(output_path)
    plt.close()

def plot_cluster_heatmaps(cof_clustered, output_path):
    clusters = cof_clustered['judge_cluster'].unique()
    clusters.sort()
    _, axes = plt.subplots(nrows=1, ncols=clusters.shape[0], figsize=(15, 5), sharey=True, gridspec_kw=dict(width_ratios=[0.8,1]))
    plt.subplots_adjust(wspace=0.05)
    params = BREWING_PARAM_CONTINUOUS
    cata = CATA
    for i, cluster in enumerate(clusters):
        cof_cluster = cof_clustered[cof_clustered['judge_cluster'] == cluster]
        data_ = cof_cluster.groupby('Brew')[params+cata+['Liking']].mean()
        data = data_[params+cata]
        data = pd.concat([data, data_['Liking']], axis=1)

        # p-value annotations
        # rho = data.corr(method='spearman')
        # pval = data.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*rho.shape)
        # p = pval.applymap(lambda x: round(x, 2))#''.join(['*' for t in [.05, .01, .001] if x<=t]))
        # p = p[cata + ['Liking']].loc[params+['Liking']]
        # rho.round(2).astype(str) + p
        # data = rho[cata + ['Liking']].loc[params+['Liking']]

        data = data.corr(method='spearman')[cata + ['Liking']].loc[params+['Liking']]

        data = data.rename(columns={'TDS__1' : 'Total Dissolved Solids'})
        data = data.rename(index={'TDS__1' : 'Total Dissolved Solids'})
        cbar = False if i == 0 else True
        sns.heatmap(data,
                    # annot=p,
                    annot=False,
                    fmt="",
                    cmap='coolwarm',
                    cbar=cbar,
                    vmin=-1, vmax=1,
                    annot_kws={"size": 6},
                    cbar_kws={"label": "Spearman correlation"},
                    linewidths=2, linecolor='white',

                    ax=axes[i])
        axes[i].set_title(f"$\\bf{{Cluster\\enspace{cluster}}}$")


        axes[i].set_xticklabels(axes[i].get_xticklabels(), fontsize=12)
    plt.savefig(output_path)
    plt.close()

def plot_clustermap(judge_clusters, output_path):
    judge_clusters = judge_clusters.sort_values(by='judge_cluster')
    colors = sns.color_palette("pastel")
    color_map = {c:colors[c+1] for c in judge_clusters['judge_cluster'].unique()}
    row_colors_series = judge_clusters['judge_cluster'].map(color_map)
    row_colors_series.rename('$\\bf{Cluster}$', inplace=True)

    sns.clustermap(judge_clusters.drop(columns='judge_cluster'),
                row_cluster=False, method='ward', cmap='coolwarm',
                vmin=-1, vmax=1, row_colors=row_colors_series,
                yticklabels=False,
                cbar_kws={"label": "Spearman correlation\nwith Liking"},
                cbar_pos=(1.0, 0.28, 0.03, 0.4))
    plt.savefig(output_path)
    plt.close()

def _plot_venn(sets, labels, title, output_path):
    venn = venn2(sets, set_labels=labels, set_colors=sns.color_palette('Set2')[:2], alpha=0.75)
    for x in range(1, 2**len(sets)):
        selection = f"{x:02b}" if len(sets) == 2 else f"{x:03b}"
        subset_label = venn.get_label_by_id(selection)
        intersection_set = [sets[i] for i, c in enumerate(selection) if c == '1']
        difference_set = [sets[i] for i, c in enumerate(selection) if c == '0']
        result_set = set.intersection(*intersection_set) - set().union(*difference_set)
        subset_label.set_text("\n".join(result_set))
        subset_label.set_color('red')
        subset_label.set_fontsize(12)
        subset_label.set_fontweight('bold')
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def plot_cluster_venn(judge_clusters, output_path):
    avg_clust_scores = judge_clusters.groupby('judge_cluster').mean()
    avg_clust_scores = avg_clust_scores[abs(avg_clust_scores) > 0.15]
    c1 = avg_clust_scores.iloc[0].dropna()
    c2 = avg_clust_scores.iloc[1].dropna()
    c1 = c1.rename({'Dark.chocolate':'Dark chocolate', 'Mouthfeel (adj)':'Balanced mouthfeel', 'Acidity (adj)':'Balanced acidity', 'Flavor.intensity (adj)':'Balanced flavor intensity'})
    c2 = c2.rename({'Tea.floral':'Tea/floral', 'Mouthfeel (adj)':'Balanced mouthfeel', 'Acidity (adj)':'Balanced acidity', 'Flavor.intensity (adj)':'Balanced flavor intensity'})
    c1 = c1.sort_values(key=abs, ascending=False)
    c2 = c2.sort_values(key=abs, ascending=False)
    _plot_venn([set(c1.index), set(c2.index)], ['Cluster 1', 'Cluster 2'], 'Drivers of liking', output_path)

def plot_cluster_feature_liking_distribution(judge_clusters, output_path):
    pairs=[(1,2)]
    features = judge_clusters.columns.difference(['judge_cluster'])
    ncols = np.sqrt(len(features)).astype(int)
    nrows = np.ceil(len(features) / ncols).astype(int)
    _, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8))
    for feature in features:
        coord = np.unravel_index(features.get_loc(feature), (nrows, ncols))
        sns.violinplot(data=judge_clusters, x='judge_cluster', y=feature, ax=axes[coord])
        axes[coord].set_xlabel("Cluster")
        axes[coord].set_ylabel(feature + '\nLiking')
        axes[coord].set_ylim([-1,1])
        annotator = Annotator(axes[coord], pairs, data=judge_clusters, x='judge_cluster', y=feature)
        annotator.configure(test='Kruskal', text_format='star', loc='outside', verbose=0)
        annotator.apply_and_annotate()
    for ax in axes.flat[len(features):]:
        ax.remove()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_cluster_brew_liking_distribution(cof_clustered, cluster_name, output_path):
    fig, axs = plt.subplots(5, 6, figsize=(10, 10))
    for i, brew in enumerate(cof_clustered['Brew'].unique()):
        cof_brew = cof_clustered[cof_clustered['Brew'] == brew]
        x=cluster_name
        y='Liking'
        coord = np.unravel_index(i, (5, 6))
        ax = axs[coord]
        sns.boxplot(data=cof_brew, x=x, y=y, ax=ax)
        pairs=[(1,2)]
        annotator = Annotator(ax, pairs, data=cof_brew, x=x, y=y)
        annotator.configure(test='Kruskal', text_format='star', loc='outside', verbose=0)
        annotator.apply_and_annotate()
        ax.set_ylabel(f"{brew} Liking")
    for ax in axs.flat[len(cof_clustered['Brew'].unique()):]:
        ax.remove()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_response_surfaces(cof, cof_features_brew, cluster_name, output_path):
    data = cof_features_brew[['TDS__1', 'Percent Extraction', 'liking']]
    # Unclear which brew was removed in Cotter paper, but seems likely it is the
    # same one selected by isolation forest since results match and the removed
    # brew appears to have very high PE and TDS
    iforest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    outliers = iforest.fit_predict(data[['TDS__1', 'Percent Extraction']])
    data = data.loc[outliers != -1]

    data_all = data.copy()
    data_c1 = cof[cof[cluster_name] == 1]
    data_c1 = data_c1.groupby('Brew')[['TDS__1', 'Percent Extraction', 'Liking']].mean()
    data_c1.rename(columns={'Liking':'liking'}, inplace=True)
    data_c2 = cof[cof[cluster_name] == 2]
    data_c2 = data_c2.groupby('Brew')[['TDS__1', 'Percent Extraction', 'Liking']].mean()
    data_c2.rename(columns={'Liking':'liking'}, inplace=True)
    data_c1 = data_c1.loc[outliers != -1]
    data_c2 = data_c2.loc[outliers != -1]
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4), layout='constrained')
    titles = ['Cluster\\,1', 'Cluster\\,2', 'Entire\\,Sample']
    for i, data in enumerate([data_c1, data_c2, data_all]):
        X = data[['Percent Extraction', 'TDS__1']].to_numpy()
        y = data['liking'].to_numpy()

        poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        poly_model.fit(X, y)

        x_grid = np.linspace(14, 28, 100)
        y_grid = np.linspace(0.9, 1.7, 100)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        # Reshape for prediction
        Z = np.c_[X_mesh.ravel(), Y_mesh.ravel()]

        Z_pred = poly_model.predict(Z)
        Z_pred = Z_pred.reshape(X_mesh.shape) # Reshape back to grid for plotting

        p = axes[i].contourf(X_mesh, Y_mesh, Z_pred, levels=1000, cmap=sns.color_palette("viridis", as_cmap=True))
        p = axes[i].contour(X_mesh, Y_mesh, Z_pred, levels=20, linewidths=0.5, alpha=0.5, cmap='Greys')
        axes[i].scatter(data['Percent Extraction'], data['TDS__1'], c=data['liking'], cmap='viridis')

        # Original BCC lines for reference
        axes[i].axvline(x=18, color='black', linestyle='--', lw=0.5)
        axes[i].axvline(x=22, color='black', linestyle='--', lw=0.5)
        axes[i].axhline(y=1.15, color='black', linestyle='--', lw=0.5)
        axes[i].axhline(y=1.35, color='black', linestyle='--', lw=0.5)

        axes[i].set_title(f"$\\bf{{{titles[i]}}}$")

    # Add colorbar
    norm = plt.Normalize(data['liking'].min(), data['liking'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='liking')
    fig.supxlabel('Percent Extraction')
    fig.supylabel('TDS')
    plt.savefig(output_path)
    plt.close()

def main():
    cof = pd.read_csv('data/cotter_dataset.csv')
    cof_features = pd.read_csv('data/processed/cof_features.csv')
    cof_clustered = pd.read_csv('data/processed/cof_with_judge_clusters.csv')
    cof_features_brew = pd.read_csv('data/processed/cof_features_brew.csv')
    cof_features_judge = pd.read_csv('data/processed/cof_features_judge.csv')
    cof_stats = pd.read_csv('data/processed/cof_feature_stats.csv', index_col=0)
    cof_stats_brew = pd.read_csv('data/processed/cof_brew_feature_stats.csv', index_col=0)
    cof_stats_judge = pd.read_csv('data/processed/cof_judge_feature_stats.csv', index_col=0)
    stats_judges = pd.read_csv('data/processed/stats_for_each_judge.csv', index_col=0)
    plot_drivers_of_liking(cof_features, cof_features_brew, cof_features_judge,
                           cof_stats, cof_stats_brew, cof_stats_judge,
                           'figures/main/dol.svg')
    # cof_jar = cof[['Flavor.intensity', 'Acidity', 'Mouthfeel']].copy()
    # plot_jar_features(cof_jar, cof['Liking'], 'figures/main/jar.svg')

    # cof_jar['judge_cluster'] = cof_clustered['judge_cluster']
    # for cluster in cof_jar['judge_cluster'].unique():
    #     cof_cluster = cof_jar[cof_jar['judge_cluster'] == cluster]
    #     cluster_liking = cof_clustered[cof_clustered['judge_cluster'] == cluster]['Liking']
    #     plot_jar_features(cof_cluster[['Flavor.intensity', 'Acidity', 'Mouthfeel']],
    #                       cluster_liking,
    #                     f'figures/supplementary/jar_cluster_{cluster}.svg')
    
    # plot_liking_corr(cof_stats, cof_stats_brew, cof_stats_judge, 'figures/main/liking_correlation.svg')
    plot_cata_corrmap(cof_features_brew[CATA+['liking']], 'figures/main/cata_correlation_map.svg')
    # plot_cluster_heatmaps(cof_clustered, 'figures/main/cluster_heatmaps.svg')
    # clusters = cof_clustered.groupby('Judge')['judge_cluster'].first()
    # stats_judges.loc[stats_judges['SRC pval'].isna(), 'SRC'] = 0
    # judge_clusters = stats_judges.pivot_table(index='Judge', columns=stats_judges.index, values='SRC')
    # judge_clusters = pd.merge(judge_clusters, clusters, left_index=True, right_index=True)
    # plot_clustermap(judge_clusters, 'figures/main/clustermap.svg')
    # plot_cluster_venn(judge_clusters, 'figures/main/clustermap.svg')
    # plot_cluster_feature_liking_distribution(judge_clusters, 'figures/supplementary/cluster_feature_liking_distribution.svg')
    # plot_cluster_brew_liking_distribution(cof_clustered, 'judge_cluster', 
    #                                       'figures/supplementary/cluster_brew_liking_distribution.svg')
    # cotter_clusters = cof_clustered.groupby('Judge')['Cluster'].first()
    # stats_judges.loc[stats_judges['SRC pval'].isna(), 'SRC'] = 0
    # judge_clusters = stats_judges.pivot_table(index='Judge', columns=stats_judges.index, values='SRC')
    # judge_clusters = pd.merge(judge_clusters, cotter_clusters, left_index=True, right_index=True)
    # judge_clusters = judge_clusters.rename(columns={'Cluster':'judge_cluster'})
    # plot_cluster_feature_liking_distribution(judge_clusters, 'figures/supplementary/cluster_feature_liking_distribution_cotter.svg')
    # plot_cluster_brew_liking_distribution(cof_clustered, 'Cluster', 
    #                                       'figures/supplementary/cluster_brew_liking_distribution_cotter.svg')
    # plot_response_surfaces(cof_clustered, cof_features_brew, 'judge_cluster',
    #                         'figures/supplementary/response_surfaces.png')
    # plot_response_surfaces(cof_clustered, cof_features_brew, 'Cluster',
    #                         'figures/supplementary/response_surfaces_cotter.png')
if __name__ == "__main__":
    main()