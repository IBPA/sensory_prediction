from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, RFE
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
import pandas as pd

from .constants import *

def generate_stat_table(X, y, discrete_features, discrete_label, model, include_rfe=True):
    # Pearson's r and p-val
    pear = X.apply(lambda x: pearsonr(x, y.loc[X.index]))
    pear = pear.T.sort_values(by=0, key=abs, ascending=False).reset_index().reset_index().set_index('index')

    # Spearman's rho and p-val
    spear = X.apply(lambda x: spearmanr(x, y.loc[X.index]))
    spear = spear.T.sort_values(by=0, key=abs, ascending=False).reset_index().reset_index().set_index('index')

    # Mutual information
    if discrete_label:
        mi = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=42)
        mi = pd.DataFrame(mi, index=X.columns)
        mi = mi.sort_values(by=0, key=abs, ascending=False).reset_index().reset_index().set_index('index')
    else:
        mi = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=42)
        mi = pd.DataFrame(mi, index=X.columns)
        mi = mi.sort_values(by=0, key=abs, ascending=False).reset_index().reset_index().set_index('index')

    # Recursive feature elimination rankings
    if include_rfe:
        rfe = RFE(estimator=model, n_features_to_select=1, step=1)
        rfe.fit(X, y)
        rfe_rank = pd.Series(rfe.ranking_, index=X.columns)

        stats = pd.concat([pear, spear, mi, rfe_rank], axis=1)

        stats.columns = ['PCC Rank', 'PCC', 'PCC pval', 'SRC Rank', 'SRC', 'SRC pval', 'MI Rank', 'MI', 'RFE Rank']
    else:
        stats = pd.concat([pear, spear, mi], axis=1)
        stats.columns = ['PCC Rank', 'PCC', 'PCC pval', 'SRC Rank', 'SRC', 'SRC pval', 'MI Rank', 'MI']

    stats['PCC Rank'] = stats['PCC Rank'].astype(int) + 1
    stats['SRC Rank'] = stats['SRC Rank'].astype(int) + 1
    stats['MI Rank'] = stats['MI Rank'].astype(int) + 1

    stats['Avg Rank'] = stats.filter(like='Rank', axis=1).mean(axis=1)
    stats.sort_values(by='Avg Rank')

    return stats

def main():
    cof_features = pd.read_csv('data/processed/cof_features.csv')
    cof_features_brew = pd.read_csv('data/processed/cof_features_brew.csv')
    cof_features_judge = pd.read_csv('data/processed/cof_features_judge.csv')
    cof_clusters = pd.read_csv('data/processed/cof_with_judge_clusters.csv')['judge_cluster']
    cof_clustered = pd.concat([cof_features, cof_clusters], axis=1)
    cof_c1 = cof_clustered[cof_clustered['judge_cluster'] == 1]
    cof_c2 = cof_clustered[cof_clustered['judge_cluster'] == 2]
    cof_c1 = cof_c1.drop(columns='judge_cluster')
    cof_c2 = cof_c2.drop(columns='judge_cluster')

    model = GradientBoostingRegressor(random_state=42)

    rating_cols = cof_features.columns.str.contains('liking')
    X = cof_features[cof_features.columns[~rating_cols]]
    y = cof_features['liking']
    discrete_features = ~X.columns.isin(BREWING_PARAM_CONTINUOUS)
    stats = generate_stat_table(X, y, discrete_features, True, model)

    rating_cols = cof_features_brew.columns.str.contains('liking')
    X = cof_features_brew[cof_features_brew.columns[~rating_cols]]
    y = cof_features_brew['liking']
    discrete_features = X.columns.isin(BREWING_PARAM_CATEGORICAL)
    stats_brew = generate_stat_table(X, y, discrete_features, False, model)

    rating_cols = cof_features_judge.columns.str.contains('liking')
    X = cof_features_judge[cof_features_judge.columns[~rating_cols]]
    y = cof_features_judge['liking']
    stats_judge = generate_stat_table(X, y, False, False, model)

    rating_cols = cof_c1.columns.str.contains('liking')
    X = cof_c1[cof_c1.columns[~rating_cols]]
    y = cof_c1['liking']
    discrete_features = ~X.columns.isin(BREWING_PARAM_CONTINUOUS)
    stats_c1 = generate_stat_table(X, y, discrete_features, True, model)

    rating_cols = cof_c2.columns.str.contains('liking')
    X = cof_c2[cof_c2.columns[~rating_cols]]
    y = cof_c2['liking']
    discrete_features = ~X.columns.isin(BREWING_PARAM_CONTINUOUS)
    stats_c2 = generate_stat_table(X, y, discrete_features, True, model)

    stats.to_csv('data/processed/cof_feature_stats.csv')
    stats_brew.to_csv('data/processed/cof_brew_feature_stats.csv')
    stats_judge.to_csv('data/processed/cof_judge_feature_stats.csv')
    stats_c1.to_csv('data/processed/cof_cluster1_feature_stats.csv')
    stats_c2.to_csv('data/processed/cof_cluster2_feature_stats.csv')

if __name__ == "__main__":
    main()