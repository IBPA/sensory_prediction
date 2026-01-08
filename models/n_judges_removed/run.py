import glob 
import re

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

cof = pd.read_csv("data/cotter_dataset.csv")
judges = cof['Judge']

cof_features = pd.read_csv("data/processed/cof_features.csv")

cof_stats = pd.read_csv("data/processed/cof_feature_stats.csv", index_col=0)
cof_stats.sort_values(by='Avg Rank', inplace=True)

# # Randomly sampled judges removed
cof_features_judges = pd.concat([cof_features, judges], axis=1)
num_judges = judges.max()
for x in range(10):
    for n in range(5, 61, 5):
        rm = judges.sample(n, random_state=x)
        subset = cof_features_judges[~cof_features_judges['Judge'].isin(rm)]
        subset = subset.drop(columns='Judge')

        selected = subset[cof_stats.index[:3].to_list()]
        X_train, X_test, y_train, y_test = train_test_split(selected, subset['binary_liking'], test_size=0.2, random_state=42)
        pd.concat([X_train, y_train], axis=1).to_csv(f'data/processed/n_judges_removed/cof_selected_train_clf_noagg_n{n}_{x}.csv', index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(f'data/processed/n_judges_removed/cof_selected_holdout_clf_noagg_n{n}_{x}.csv', index=False)
        pd.concat([selected, subset['binary_liking']], axis=1).to_csv(f'data/processed/n_judges_removed/cof_selected_all_clf_noagg_n{n}_{x}.csv', index=False)

        selected = subset[cof_stats.index[:15].to_list()]
        X_train, X_test, y_train, y_test = train_test_split(selected, subset['liking'], test_size=0.2, random_state=42)
        pd.concat([X_train, y_train], axis=1).to_csv(f'data/processed/n_judges_removed/cof_selected_train_reg_noagg_n{n}_{x}.csv', index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(f'data/processed/n_judges_removed/cof_selected_holdout_reg_noagg_n{n}_{x}.csv', index=False)
        pd.concat([selected, subset['binary_liking']], axis=1).to_csv(f'data/processed/n_judges_removed/cof_selected_all_reg_noagg_n{n}_{x}.csv', index=False)




