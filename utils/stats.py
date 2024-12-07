import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint

def cramers_v(x, y):
    """
    Compute Cramér's V for two categorical columns.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    # Correct phi2 for bias when r or k > 1
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))

def chi_square_test(data, column, target):
    contingency_table = pd.crosstab(data[column], data[target])
    _, p_value, _, _ = chi2_contingency(contingency_table)
    return p_value


def calculate_confidence_intervals(contingency_table, alpha=0.05):
    """
    Calculate confidence intervals for proportions based on a contingency table.

    Parameters:
        contingency_table (pd.DataFrame): Contingency table with groups as index and TARGET values as columns.
        alpha (float): Significance level for confidence interval. Default is 0.05 (95% CI).

    Returns:
        pd.DataFrame: DataFrame containing group, proportion, and confidence intervals.
    """
    confidence_intervals = []
    for group in contingency_table.index:
        # Total samples in the group
        total = contingency_table.loc[group].sum()
        # Defaults in the group (assumes 1 is the column for defaults)
        defaults = contingency_table.loc[group, 1]
        # Proportion of defaults
        prop = defaults / total
        # Compute confidence interval
        ci_low, ci_high = proportion_confint(defaults, total, alpha=alpha, method='normal')
        confidence_intervals.append((group, prop, ci_low, ci_high))

    # Convert to DataFrame
    ci_df = pd.DataFrame(confidence_intervals, columns=['Group', 'Proportion', 'CI Lower', 'CI Upper'])
    return ci_df

def plot_confidence_intervals(ci_df, xlabel='Group', ylabel='Proportion of '
                                                            'Defaults ('
                                                            'TARGET=1)',
                              title='Confidence Intervals', figsize=(4, 3)):
    """
    Plot confidence intervals for proportions.

    Parameters:
        ci_df (pd.DataFrame): DataFrame containing group, proportion, and confidence intervals.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title for the plot.
    """
    plt.figure(figsize=figsize)
    plt.errorbar(ci_df['Group'], ci_df['Proportion'],
                 yerr=[ci_df['Proportion'] - ci_df['CI Lower'], ci_df['CI Upper'] - ci_df['Proportion']],
                 fmt='o', capsize=5, label='95% CI')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

