import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import click

from .constants import *

# Sequential feature selection based on computed feature importance rankings
def feature_selection(X, y, sorted_features, model, scoring, stratified=True, n_splits=10):
    avg_scores = []
    all_scores = {}
    for i, feature in tqdm(enumerate(sorted_features.index), total=len(sorted_features)):
        feature_subset = X[sorted_features[:i+1].index]
        if feature_subset.shape[1] == 0: continue
        # 5 fold cv
        scores = []
        kfold = StratifiedKFold(n_splits=n_splits) if stratified else KFold(n_splits=n_splits)
        for train_index, test_index in kfold.split(feature_subset, y):
            X_train, X_test = feature_subset.iloc[train_index], feature_subset.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train.to_numpy().ravel())
            y_pred = model.predict(X_test)
            scores.append(scoring(y_test, y_pred))
        all_scores[i+1] = scores
        avg_scores.append(np.mean(scores))
    
    return pd.DataFrame(all_scores).melt(var_name='n_features', value_name='score') 

def do_rfe(X, y, model, scoring, output_path):
    cv_strategy = LeaveOneOut()
    rfecv = RFECV(estimator=model, min_features_to_select=1, step=1,
                    cv=cv_strategy, scoring=scoring, verbose=0, n_jobs=-1)
    rfecv.fit(X, y)

    # Borrowed from the sklearn documentation
    data = {
        key: value
        for key, value in rfecv.cv_results_.items()
        if key in ["n_features", "mean_test_score", "std_test_score"]
    }
    cv_results = pd.DataFrame(data)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel(scoring.replace('_', ' ').title())
    plt.errorbar(
        x=cv_results["n_features"],
        y=cv_results["mean_test_score"],
        yerr=cv_results["std_test_score"],
    )
    plt.title("Recursive Feature Elimination")
    plt.savefig(output_path)
    plt.close()

def plot_feature_selection(scores, selection, n_features, name):
    sns.lineplot(data=scores, x='n_features', y='score')
    sns.scatterplot(data=scores.groupby('n_features').mean(), x='n_features', y='score')
    highest_score = scores.groupby('n_features').mean()['score'].idxmax()
    plt.axvline(x=highest_score, color='red', linestyle='--', label='Highest')
    plt.axvline(x=selection, color='blue', linestyle='--', label='Selection')
    plt.legend(loc='upper right')
    # Legend adjust
    plt.xlabel('Number of features')
    plt.ylabel('Average score')
    plt.title('Feature selection')
    plt.xticks(range(1, n_features+1, 5))
    plt.savefig(f'figures/supplementary/{name}.svg')
    plt.close()

def save_data(data, selected_features, name):
    data = data[selected_features]
    data.to_csv(f'data/processed/{name}.csv', index=False)


@click.command()
@click.option(
    '--mode',
    type=str,
    default='run',
)
def main(mode):
    cof = pd.read_csv("data/cotter_dataset.csv")
    judges = cof['Judge']
    cof_features = pd.read_csv("data/processed/cof_features.csv")
    cof_features_brew = pd.read_csv("data/processed/cof_features_brew.csv")
    cof_features_judge = pd.read_csv("data/processed/cof_features_judge.csv")

    cof_clusters = pd.read_csv('data/processed/cof_with_judge_clusters.csv')['judge_cluster']
    cof_clustered = pd.concat([cof_features, cof_clusters], axis=1)
    cof_c1 = cof_clustered[cof_clustered['judge_cluster'] == 1]
    cof_c2 = cof_clustered[cof_clustered['judge_cluster'] == 2]
    cof_c1 = cof_c1.drop(columns='judge_cluster')
    cof_c2 = cof_c2.drop(columns='judge_cluster')

    cof_stats = pd.read_csv("data/processed/cof_feature_stats.csv", index_col=0)
    cof_stats.sort_values(by='Avg Rank', inplace=True)
    cof_stats_brew = pd.read_csv("data/processed/cof_brew_feature_stats.csv", index_col=0)
    cof_stats_brew.sort_values(by='Avg Rank', inplace=True)
    cof_stats_judge = pd.read_csv("data/processed/cof_judge_feature_stats.csv", index_col=0)
    cof_stats_judge.sort_values(by='Avg Rank', inplace=True)
    cof_stats_c1 = pd.read_csv("data/processed/cof_cluster1_feature_stats.csv", index_col=0)
    cof_stats_c1.sort_values(by='Avg Rank', inplace=True)
    cof_stats_c2 = pd.read_csv("data/processed/cof_cluster2_feature_stats.csv", index_col=0)
    cof_stats_c2.sort_values(by='Avg Rank', inplace=True)
    

    regressor_model = GradientBoostingRegressor(random_state=42)
    clf_model = RandomForestClassifier(random_state=42)
    
    if mode == 'run':
        # Non-aggregated features
        rating_cols = cof_features.columns.str.contains('liking')
        X = cof_features[cof_features.columns[~rating_cols]]
        y = cof_features['liking']
        
        scores  = feature_selection(X, y, cof_stats, regressor_model, r2_score)
        scores.to_csv('data/processed/feature_selection_scores_reg.csv', index=False)
        y = cof_features['binary_liking']
        scores = feature_selection(X, y, cof_stats, clf_model, f1_score)
        scores.to_csv('data/processed/feature_selection_scores_clf.csv', index=False)

        # Clusters
        # Cluster 1
        rating_cols = cof_c1.columns.str.contains('liking')
        X = cof_c1[cof_c1.columns[~rating_cols]]
        y = cof_c1['liking']
        scores  = feature_selection(X, y, cof_stats_c1, regressor_model, r2_score, False)
        scores.to_csv('data/processed/c1_feature_selection_scores_reg.csv', index=False)
        y = cof_c1['binary_liking']
        scores = feature_selection(X, y, cof_stats_c1, clf_model, f1_score)
        scores.to_csv('data/processed/c1_feature_selection_scores_clf.csv', index=False)
        # Cluster 2
        rating_cols = cof_c2.columns.str.contains('liking')
        X = cof_c2[cof_c2.columns[~rating_cols]]
        y = cof_c2['liking']
        scores  = feature_selection(X, y, cof_stats_c2, regressor_model, r2_score, False)
        scores.to_csv('data/processed/c2_feature_selection_scores_reg.csv', index=False)
        y = cof_c2['binary_liking']
        scores = feature_selection(X, y, cof_stats_c2, clf_model, f1_score)
        scores.to_csv('data/processed/c2_feature_selection_scores_clf.csv', index=False)

        # Brew aggregated features
        X = cof_features_brew[cof_features_brew.columns[~rating_cols]]
        y = cof_features_brew['liking']
        scores  = feature_selection(X, y, cof_stats_brew, regressor_model, r2_score, False)
        scores.to_csv('data/processed/brew_feature_selection_scores_reg.csv', index=False)
        y = cof_features_brew['binary_liking']
        scores = feature_selection(X, y, cof_stats_brew, clf_model, f1_score)
        scores.to_csv('data/processed/brew_feature_selection_scores_clf.csv', index=False)

        # Judge aggregated features
        rating_cols = cof_features_judge.columns.str.contains('liking')
        X = cof_features_judge[cof_features_judge.columns[~rating_cols]]
        y = cof_features_judge['liking']
        scores  = feature_selection(X, y, cof_stats_judge, regressor_model, r2_score, False)
        scores.to_csv('data/processed/judge_feature_selection_scores_reg.csv', index=False)
        y = cof_features_judge['binary_liking']
        scores = feature_selection(X, y, cof_stats_judge, clf_model, f1_score)
        scores.to_csv('data/processed/judge_feature_selection_scores_clf.csv', index=False)

        # Brew aggregation params only RFE
        params_only = cof_features_brew[BREWING_PARAM_CONTINUOUS]
        do_rfe(params_only, cof_features_brew['binary_liking'], 
               clf_model, 'accuracy',
               output_path='figures/supplementary/rfe_params_only_brew_clf.svg')
        do_rfe(params_only, cof_features_brew['liking'], 
               regressor_model, 'neg_mean_squared_error',
               output_path='figures/supplementary/rfe_params_only_brew_reg.svg')
    
    if mode == 'plot':
        scores = pd.read_csv('data/processed/feature_selection_scores_reg.csv')
        plot_feature_selection(scores, selection=15, 
                               n_features=scores['n_features'].max(), 
                               name='feature_selection_reg_noagg')

        scores = pd.read_csv('data/processed/feature_selection_scores_clf.csv')
        plot_feature_selection(scores, selection=3, 
                               n_features=scores['n_features'].max(),
                               name='feature_selection_clf_noagg')
        
        scores = pd.read_csv('data/processed/c1_feature_selection_scores_reg.csv')
        plot_feature_selection(scores, selection=15, 
                               n_features=scores['n_features'].max(), 
                               name='feature_selection_reg_noagg_c1')
        
        scores = pd.read_csv('data/processed/c2_feature_selection_scores_reg.csv')
        plot_feature_selection(scores, selection=15, 
                               n_features=scores['n_features'].max(), 
                               name='feature_selection_reg_noagg_c2')
        
        
        scores = pd.read_csv('data/processed/c1_feature_selection_scores_clf.csv')
        plot_feature_selection(scores, selection=3, 
                               n_features=scores['n_features'].max(), 
                               name='feature_selection_clf_noagg_c1')
        
        scores = pd.read_csv('data/processed/c2_feature_selection_scores_clf.csv')
        plot_feature_selection(scores, selection=3, 
                               n_features=scores['n_features'].max(), 
                               name='feature_selection_clf_noagg_c2')

        scores = pd.read_csv('data/processed/brew_feature_selection_scores_reg.csv')
        plot_feature_selection(scores, selection=10, 
                               n_features=scores['n_features'].max(),
                               name='feature_selection_reg_brew')

        scores = pd.read_csv('data/processed/brew_feature_selection_scores_clf.csv')
        plot_feature_selection(scores, selection=4, 
                               n_features=scores['n_features'].max(),
                               name='feature_selection_clf_brew')

        scores = pd.read_csv('data/processed/judge_feature_selection_scores_reg.csv')
        plot_feature_selection(scores, selection=9, 
                               n_features=scores['n_features'].max(),
                               name='feature_selection_reg_judge')

        scores = pd.read_csv('data/processed/judge_feature_selection_scores_clf.csv')
        plot_feature_selection(scores, selection=3, 
                               n_features=scores['n_features'].max(),
                               name='feature_selection_clf_judge')

    if mode == 'save':
        # No aggregation
        # Regression
        selected = cof_features[cof_stats.index[:15].to_list()]
        X_train, X_test, y_train, y_test = train_test_split(selected, cof_features['liking'], test_size=0.2, random_state=42)
        pd.concat([X_train, y_train], axis=1).to_csv('data/processed/cof_selected_train_reg_noagg.csv', index=False)
        pd.concat([X_test, y_test], axis=1).to_csv('data/processed/cof_selected_holdout_reg_noagg.csv', index=False)
        pd.concat([selected, cof_features['liking']], axis=1).to_csv('data/processed/cof_selected_all_reg_noagg.csv', index=False)

        # Classification
        selected = cof_features[cof_stats.index[:3].to_list()]
        X_train, X_test, y_train, y_test = train_test_split(selected, cof_features['binary_liking'], test_size=0.2, random_state=42)
        pd.concat([X_train, y_train], axis=1).to_csv('data/processed/cof_selected_train_clf_noagg.csv', index=False)
        pd.concat([X_test, y_test], axis=1).to_csv('data/processed/cof_selected_holdout_clf_noagg.csv', index=False)
        pd.concat([selected, cof_features['binary_liking']], axis=1).to_csv('data/processed/cof_selected_all_clf_noagg.csv', index=False)

        # Clustered
        # Regression
        # Cluster 1
        selected = cof_c1[cof_stats_c1.index[:15].to_list()]
        X_train, X_test, y_train, y_test = train_test_split(selected, cof_c1['liking'], test_size=0.2, random_state=42)
        pd.concat([X_train, y_train], axis=1).to_csv('data/processed/cof_selected_train_reg_noagg_c1.csv', index=False)
        pd.concat([X_test, y_test], axis=1).to_csv('data/processed/cof_selected_holdout_reg_noagg_c1.csv', index=False)
        pd.concat([selected, cof_c1['liking']], axis=1).to_csv('data/processed/cof_selected_all_reg_noagg_c1.csv', index=False)
        # Cluster 2
        selected = cof_c2[cof_stats_c2.index[:15].to_list()]
        X_train, X_test, y_train, y_test = train_test_split(selected, cof_c2['liking'], test_size=0.2, random_state=42)
        pd.concat([X_train, y_train], axis=1).to_csv('data/processed/cof_selected_train_reg_noagg_c2.csv', index=False)
        pd.concat([X_test, y_test], axis=1).to_csv('data/processed/cof_selected_holdout_reg_noagg_c2.csv', index=False)
        pd.concat([selected, cof_c2['liking']], axis=1).to_csv('data/processed/cof_selected_all_reg_noagg_c2.csv', index=False)

        # Classification
        # Cluster 1
        selected = cof_c1[cof_stats_c1.index[:3].to_list()]
        X_train, X_test, y_train, y_test = train_test_split(selected, cof_c1['binary_liking'], test_size=0.2, random_state=42)
        pd.concat([X_train, y_train], axis=1).to_csv('data/processed/cof_selected_train_clf_noagg_c1.csv', index=False)
        pd.concat([X_test, y_test], axis=1).to_csv('data/processed/cof_selected_holdout_clf_noagg_c1.csv', index=False)
        pd.concat([selected, cof_c1['binary_liking']], axis=1).to_csv('data/processed/cof_selected_all_clf_noagg_c1.csv', index=False)
        # Cluster 2
        selected = cof_c2[cof_stats_c2.index[:3].to_list()]
        X_train, X_test, y_train, y_test = train_test_split(selected, cof_c2['binary_liking'], test_size=0.2, random_state=42)
        pd.concat([X_train, y_train], axis=1).to_csv('data/processed/cof_selected_train_clf_noagg_c2.csv', index=False)
        pd.concat([X_test, y_test], axis=1).to_csv('data/processed/cof_selected_holdout_clf_noagg_c2.csv', index=False)
        pd.concat([selected, cof_c2['binary_liking']], axis=1).to_csv('data/processed/cof_selected_all_clf_noagg_c2.csv', index=False)

        # # Randomly sampled judges removed
        cof_features_judges = pd.concat([cof_features, judges], axis=1)
        num_judges = judges.max()
        for n in range(5, num_judges//2, 5):
            rm = judges.sample(n, random_state=42)
            subset = cof_features_judges[~cof_features_judges['Judge'].isin(rm)]
            subset = subset.drop(columns='Judge')

            selected = subset[cof_stats.index[:3].to_list()]
            X_train, X_test, y_train, y_test = train_test_split(selected, subset['binary_liking'], test_size=0.2, random_state=42)
            pd.concat([X_train, y_train], axis=1).to_csv(f'data/processed/n_judges_removed/cof_selected_train_clf_noagg_{n}.csv', index=False)
            pd.concat([X_test, y_test], axis=1).to_csv(f'data/processed/n_judges_removed/cof_selected_holdout_clf_noagg_{n}.csv', index=False)
            pd.concat([selected, subset['binary_liking']], axis=1).to_csv(f'data/processed/n_judges_removed/cof_selected_all_clf_noagg_{n}.csv', index=False)

            selected = subset[cof_stats.index[:15].to_list()]
            X_train, X_test, y_train, y_test = train_test_split(selected, subset['liking'], test_size=0.2, random_state=42)
            pd.concat([X_train, y_train], axis=1).to_csv(f'data/processed/n_judges_removed/cof_selected_train_reg_noagg_{n}.csv', index=False)
            pd.concat([X_test, y_test], axis=1).to_csv(f'data/processed/n_judges_removed/cof_selected_holdout_reg_noagg_{n}.csv', index=False)
            pd.concat([selected, subset['binary_liking']], axis=1).to_csv(f'data/processed/n_judges_removed/cof_selected_all_reg_noagg_{n}.csv', index=False)



        # Brew aggregation 
        # Regression
        X = cof_features_brew[cof_stats_brew.index[:10].to_list()]
        y = cof_features_brew['liking']
        pd.concat([X, y], axis=1).to_csv('data/processed/cof_selected_reg_brew.csv', index=False)
        # Classification
        X = cof_features_brew[cof_stats_brew.index[:4].to_list()]
        y = cof_features_brew['binary_liking']
        pd.concat([X, y], axis=1).to_csv('data/processed/cof_selected_clf_brew.csv', index=False)

        # Brew params only 
        X = cof_features_brew[['TDS__1', 'pH', 'Pour Temp', 'Volume']]
        y = cof_features_brew['liking']
        pd.concat([X, y], axis=1).to_csv('data/processed/cof_selected_params_only_reg_brew.csv', index=False)
        y = (cof_features_brew['liking'] >= cof_features_brew['liking'].mean()).astype(int)
        y.rename('binary_liking', inplace=True)
        pd.concat([X, y], axis=1).to_csv('data/processed/cof_selected_params_only_clf_brew.csv', index=False)
        
        # Judge aggregation 
        # Regression
        selected = cof_features_judge[cof_stats_judge.index[:9].to_list()]
        X_train, X_test, y_train, y_test = train_test_split(selected, cof_features_judge['liking'], test_size=0.2, random_state=42)
        pd.concat([X_train, y_train], axis=1).to_csv('data/processed/cof_selected_train_reg_judge.csv', index=False)
        pd.concat([X_test, y_test], axis=1).to_csv('data/processed/cof_selected_holdout_reg_judge.csv', index=False)
        # Classification
        selected = cof_features_judge[cof_stats_judge.index[:3].to_list()]
        X_train, X_test, y_train, y_test = train_test_split(selected, cof_features_judge['binary_liking'], test_size=0.2, random_state=42)
        pd.concat([X_train, y_train], axis=1).to_csv('data/processed/cof_selected_train_clf_judge.csv', index=False)
        pd.concat([X_test, y_test], axis=1).to_csv('data/processed/cof_selected_holdout_clf_judge.csv', index=False)

   
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()