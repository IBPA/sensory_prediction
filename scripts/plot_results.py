# -*- coding: utf-8 -*-
"""The run script for evaluating the model selection pipeline.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * Visualization functions.

"""
import os
import json

import pandas as pd
from sklearn import set_config
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import click

from .utils import load_model_pipeline, get_all_metrics, get_all_metrics_reg

set_config(transform_output='pandas')

def format_classification_output(metrics, path_outputs_dir):
    """Format the classification output."""
    print(metrics)
    print(f"{metrics['tp']}\t{metrics['fp']}\n{metrics['fn']}\t{metrics['tn']}")
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df = metrics_df.rename(
        columns={
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1 Score',
            'auroc': 'AUROC',
            'ap': 'Average Precision',
            'sensitivity': 'Sensitivity',
            'specificity': 'Specificity',
            'balanced_accuracy': 'Balanced Accuracy',
            'tp': 'True Positive',
            'fp': 'False Positive',
            'tn': 'True Negative',
            'fn': 'False Negative',
        }
    )

    metrics_df.to_csv(
        f"{path_outputs_dir}/classifiers/holdout_set_metrics_bin.csv",
        index=False,
    )
    metrics = {k : v.item() for k, v in metrics.items()}
    json.dump(
        metrics,
        open(
            f"{path_outputs_dir}/classifiers/holdout_set_metrics_bin.json",
            'w',
        ),
        indent=4,
    )

def generate_classification_plots(model, inputs, labels, path_outputs_dir):
    """Generate classification plots."""
    sns.set_theme(style='whitegrid')
    RocCurveDisplay.from_estimator(
        model, inputs, labels, name='ROC curve'
    )
    # RocCurveDisplay.figure_
    plt.savefig(
        f"{path_outputs_dir}/classifiers/roc_curve_bin_c1.svg",
        dpi=300,
        bbox_inches='tight',
    )
    PrecisionRecallDisplay.from_estimator(
        model, inputs, labels, name='Precision-Recall curve'
    )
    plt.savefig(
        f"{path_outputs_dir}/classifiers/pr_curve_bin_c1.svg",
        dpi=300,
        bbox_inches='tight',
    )

def format_regression_output(metrics, path_outputs_dir):
    """Format the regression output."""
    print(metrics)
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df = metrics_df.rename(
        columns={
            'mae': 'MAE',
            'mse': 'MSE',
            'r2': 'R2',
        }
    )
    metrics_df.to_csv(
        f"{path_outputs_dir}/regressors/holdout_set_metrics_reg.csv",
        index=False,
    )
    metrics = {k : v for k, v in metrics.items()}
    json.dump(
        metrics,
        open(
            f"{path_outputs_dir}/regressors/holdout_set_metrics_reg.json",
            'w',
        ),
        indent=4,
    )

def generate_regression_plots(model, metrics, inputs, labels, path_outputs_dir):
    """Generate regression plots."""
    sns.set_theme(style='whitegrid')
    y_pred = model.predict(inputs)
    plt.scatter(labels, y_pred, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    
    #######################
    plt.xticks(range(1,10))
    plt.yticks(range(1,10))
    #######################

    plt.annotate(
        f"MAE: {metrics['mae']:.2f}\nMSE: {metrics['mse']:.2f}\nR2: {metrics['r2']:.2f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'),
    )
    plt.plot(
        [labels.min(), labels.max()],
        [labels.min(), labels.max()],
        'k--',
        lw=2,
        label='Perfect Prediction',
        color='red',
        alpha=0.5,
    )
    plt.savefig(
        f"{path_outputs_dir}/regressors/true_vs_predicted_reg.png",
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


@click.command()
@click.argument(
    'path-data-test',
    type=click.Path(exists=True),
)
@click.argument(
    'path-outputs-dir',
    type=click.Path(exists=True),
)
@click.option(
    '--compare',
    type=str,
    default=None,
    multiple=True,
)
@click.option(
    '--column-target',
    type=str,
    default=None,
)
@click.option(
    '--override-model-index',
    type=int,
    default=None,
)
@click.option(
    '--mode',
    type=str,
    default='binary',
)
def main(
    path_data_test,
    path_outputs_dir,
    paths_comparison_dirs,
    column_target,
    override_model_index,
    mode='binary'
):
    """Main function with click interface.

    Args:
        path_data_test (str): The path to the test data.
        path_outputs_dir (str): The path to the outputs directory.
        column_target (str): The name of the target column.

    """
    for result_path in result_paths:
        results = json.load(open(result_path, 'r'))

    if mode == 'regression':
        metrics = get_all_metrics_reg(y_true, y_pred)
        format_regression_output(metrics, path_outputs_dir)
        generate_regression_plots(model, metrics, inputs, labels, path_outputs_dir)
        pd.DataFrame(y_pred).to_csv('model_predictions.csv')
    else:
        y_prob = model.predict_proba(inputs)[:, 1]
        metrics = get_all_metrics(y_true, y_pred, y_prob)
        format_classification_output(metrics, path_outputs_dir)
        generate_classification_plots(model, inputs, labels, path_outputs_dir)


    
    


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
