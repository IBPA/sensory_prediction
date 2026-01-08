import glob 
import re

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.DataFrame()
for filename in glob.glob("./models/n_judges_removed/**/holdout_set_metrics_reg.csv", recursive=True):
    match = re.search(r'output_\w+?_n(\d+)_(\d+)', filename)
    n = int(match.groups()[0])
    x = int(match.groups()[1])
    scores = pd.read_csv(filename)
    scores['n'] = n
    scores['num_judges'] = 118 - n
    scores['x'] = x
    df = pd.concat([df, scores])

metric = 'R2'
means = df.groupby('n').mean()
means['n'] = means.index
stds = df.groupby('n').std()

ax = sns.scatterplot(data=means, x='num_judges', y=metric)
ax.errorbar(
    x=means['num_judges'],
    y=means[metric],
    yerr=stds[metric],
    fmt="", 
    ecolor='black', 
    capsize=5,    
    alpha=0.7 
)
ax.set_xlabel('Number of consumers used for training')

c1_metrics = pd.read_csv("models/clustered/output_c1_reg/regressors/holdout_set_metrics_reg.csv")
c2_metrics = pd.read_csv("models/clustered/output_c2_reg/regressors/holdout_set_metrics_reg.csv")
plt.scatter(x=57, y=c1_metrics[metric], color='gold', marker='x', s=100, label='Cluster 1 Model')
plt.scatter(x=61, y=c2_metrics[metric], color='skyblue', marker='x', s=100, label='Cluster 2 Model')
plt.scatter(x=59, y=(c1_metrics[metric]+c2_metrics[metric])/2, color='green', marker='*', s=100, label='Average of Cluster 1 and Cluster 2 Models')
plt.legend()

plt.savefig('figures/supplementary/n_judges_removed.svg')
plt.close()