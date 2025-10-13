# --------------------------------------------------------------------------------------
# Full pipeline for wilcoxon test processing
#
# Uses statannotations
# install from pull request:
# pip3 install git+https://github.com/getzze/statannotations.git@compat-seaborn-13 --upgrade
# --------------------------------------------------------------------------------------
import os
import itertools
import numpy as np
import scipy.stats as stats
from src.analysis import statannotations_permutation


fontSize = 10


# --------------------------------------------------------------------------------------
# Simply avg and Std Dev printing function...
# --------------------------------------------------------------------------------------
def printAveragesAndStd(dataset):
    print("\nAverages and Std Dev:")
    if isinstance(dataset, dict):
        for setToPlot in dataset:
            print(f'{setToPlot}: avg={np.nanmean(dataset[setToPlot])}, stdev={np.nanstd(dataset[setToPlot])}, len={len(dataset[setToPlot])}')
    else:
        for pos, setToPlot in enumerate(dataset.T):
            print(f'{pos}: avg={np.nanmean(setToPlot)}, stdev={np.nanstd(setToPlot)}, len={np.count_nonzero(~np.isnan(setToPlot))}')  # ~ inverts the boolean matrix returned from np.isnan
    print()


# --------------------------------------------------------------------------------------
# This is a simple check to prevent the "All numbers are identical in mannwhitneyu" error...
# --------------------------------------------------------------------------------------
def checkTiecorrect(x,y):
    x = np.asarray(x)
    y = np.asarray(y)
    n1 = len(x)
    n2 = len(y)
    ranked = stats.rankdata(np.concatenate((x, y)))
    rankx = ranked[0:n1]  # get the x-ranks
    u1 = n1*n2 + (n1*(n1+1))/2.0 - np.sum(rankx, axis=0)  # calc U for x
    u2 = n1*n2 - u1  # remainder is U for y
    T = stats.tiecorrect(ranked)
    return T != 0



def plotMeanVars(ax, data, pos, title):
    points = [data[d] for d in data]
    positions = [pos[d] for d in data]
    ax.boxplot(points, positions=positions, labels=data.keys())  # notch='True', patch_artist=True,

    for d in data:
        ax.plot(pos[d]*np.ones(len(data[d])), np.array(data[d]).reshape(len(data[d])),
                'r.', alpha=0.2)
    ax.set_title(title)


# h = fontSize / 10
barHeight = fontSize / 2.
def plotSignificanceStars(ax, tests, pos, plotOrder = None, col='grey'):
    def stars(p):
       if p < 0.0001:
           return "****"
       elif (p < 0.001):
           return "***"
       elif (p < 0.01):
           return "**"
       elif (p < 0.05):
           return "*"
       else:
           return "-"

    # def stars(test):
    #     if test < 0.001:
    #         text = f'*** p={test:.4f}'
    #     elif test < 0.01:
    #         text = f'** p={test:.3f}'
    #     elif test < 0.05:
    #         text = f'* p={test:.3f}'
    #     else:
    #         text = f'p={test:.3f}'
    #     return text

    def plotBar(x1, x2, h, text):  # (x1, x2, y, h, text):
        ylim = ax.get_ylim()
        y = ylim[1] + h
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=col)
        ax.set_ylim([ylim[0], y + 5 * h])

    if plotOrder is None: plotOrder = tests
    # * statistical tests. From https://towardsdatascience.com/beautiful-boxplots-with-statistical-significance-annotation-e1b314927fc5
    # x1, x2 = -0.20, 0.20
    # y, h, col = df_long[df_long.Feature == feature][“Value”].max()+1, 2, ‘k’
    # axes[idx].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    # axes[idx].text((x1+x2)*.5, y+h, “statistically significant”, ha=’center’, va=’bottom’, color=col)
    ylim = ax.get_ylim()
    h = (ylim[1] - ylim[0]) / 50
    for order, pair in enumerate(plotOrder):
        labels = pair.split('_')
        plotBar(pos[labels[0]], pos[labels[1]], h, stars(tests[pair]))  # ylim[1] + delta * order
    print()


def computeWilcoxonTests(data):
    tests = {}
    for pair in itertools.combinations(data, r=2):
        if pair[0] != pair[1]:
            testName = pair[0]+'_'+pair[1]
            if checkTiecorrect(data[pair[0]], data[pair[1]]):
                tests[testName] = stats.mannwhitneyu(data[pair[0]], data[pair[1]]).pvalue
            else:
                tests[testName] = 1
            print(f'test[{testName}] = {tests[testName]}')
    return tests

# ----------------------------------------------------------------------------
# Some convenience WholeBrain
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Plotting func.
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt

posA = 1; posB = 2; posC = 3; posD = 4

# Generates a boxPlot and the p-values for 3 different labels
def plotComparisonAcrossLabelsAx(ax, dataA, dataB, dataC, labels, titleLabel='test', ylabel='Obs', yLimits = None):
    points = {labels[0]: dataA, labels[1]: dataB, labels[2]: dataC}
    positions = {labels[0]: posA, labels[1]: posB, labels[2]: posC}
    if yLimits is not None:
        ax.set_ylim(yLimits)
    plotMeanVars(ax, points, positions, title=titleLabel)  # f'Parm Comparison ({titleLabel})'
    test = computeWilcoxonTests(points)
    plotSignificanceStars(ax, test, positions, plotOrder=[labels[0]+'_'+labels[1],
                                                     labels[1]+'_'+labels[2],
                                                     labels[0]+'_'+labels[2],
                                                    ])
    ax.set_ylabel(ylabel)


# Convenience version that directly generates the picture...
def plotComparisonAcrossLabels(dataA, dataB, dataC, labels, titleLabel='test', ylabel='Obs', yLimits=None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plotComparisonAcrossLabelsAx(ax, dataA, dataB, dataC, labels, titleLabel=titleLabel, ylabel=ylabel, yLimits=yLimits)
    plt.show()


# Same as previous one, but with 4 labels. Too lazy to refactor this... ;-)
def plotValuesComparisonAcross4Labels(dataA, dataB, dataC, dataD, labels, titleLabel='test', yLimits = None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    points = {labels[0]: dataA, labels[1]: dataB, labels[2]: dataC, labels[3]: dataD}
    positions = {labels[0]: posA, labels[1]: posB, labels[2]: posC, labels[3]: posD}
    if yLimits is not None:
        ax.set_ylim(yLimits)
    plotMeanVars(ax, points, positions, title=titleLabel)  # f'Parm Comparison ({titleLabel})'
    test = computeWilcoxonTests(points)
    plotSignificanceStars(ax, test, positions, plotOrder=[labels[0]+'_'+labels[1],
                                                     labels[1]+'_'+labels[2],
                                                     labels[0]+'_'+labels[2],
                                                     labels[2]+'_'+labels[3],
                                                     labels[1]+'_'+labels[3],
                                                     labels[0]+'_'+labels[3],
                                                    ])
    ax.set_ylabel("phFCD")
    plt.show()


def findMinMaxSpan(a,b):
    max = -np.inf; posMax = 0
    min = np.inf; posMin = 0
    for pos, (va, vb) in enumerate(zip(a,b)):
        span = np.abs(va-vb)
        if span > max:
            max = span
            posMax = pos
        if span < min:
            min = span
            posMin = pos
    return min, posMin, max, posMax


# --------------------------------------------------------------------------------------
# Full pipeline using the statannotations library, much better replacement for my
# own p_values implementation...
# https://github.com/trevismd/statannotations
# --------------------------------------------------------------------------------------
from itertools import combinations
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items  # internal Pandas bugfix for Pandas 2.0
import seaborn as sns
from statannotations.Annotator import Annotator


def padEqualLengtLists(tests):
    totalLen = max([len(l) for l in tests])
    fixed = []
    for t in tests:
        fixed.append(np.pad(t, (0, totalLen-len(t)), 'constant', constant_values=np.nan))
    return fixed


def padEqualLengthDicts(tests):
    totalLen = max([len(l) for l in tests.values()])
    fixed = {}
    for c in tests:
        fixed[c] = np.pad(tests[c], (0,totalLen-len(tests[c])), 'constant', constant_values=np.nan)
    return fixed


def plotComparisonAcrossLabels2Ax(ax, tests, custom_test=None,
                                  columnLables=None, graphLabel='', pairs=None,
                                  test='Mann-Whitney', comparisons_correction='BH'):
    printAveragesAndStd(tests)
    if columnLables is None:
        columnLables = tests.keys()
    if isinstance(tests, dict):
        tests = padEqualLengthDicts(tests)
    df = pd.DataFrame(tests, columns=columnLables)
    default_color_map = {'HC': '#1f77b4', 'MCI': '#ff7f0e', 'AD': '#2ca02c'}
    tab10 = sns.color_palette('tab10')
    palette_list = [default_color_map.get(label, tab10[i % len(tab10)]) for i, label in enumerate(columnLables)]
    palette = dict(zip(columnLables, palette_list))
    sns.boxplot(data=df, order=columnLables, ax=ax, palette=palette)
    # sns.catplot(data=df, kind="box")
    if pairs == None:
        pairs = list(combinations(columnLables, 2))
    annotator = Annotator(ax, pairs, data=df, order=list(columnLables))
    if custom_test is None:
        annotator.configure(test='Mann-Whitney')
    else:
        annotator.configure(test=custom_test)
    annotator.configure(test=test, text_format='star', loc='inside')
    if comparisons_correction is not None:
        annotator.configure(comparisons_correction=comparisons_correction, correction_format="replace")  # BH / Bonferroni
    annotator.apply_and_annotate()
    ax.set_title(graphLabel)


def plotComparisonAcrossLabels2(tests, custom_test=None,
                                columnLables=None, graphLabel='', pairs=None,
                                test='Mann-Whitney', comparisons_correction='BH',save_path=None, dpi=300):
    fig, ax = plt.subplots()
    plotComparisonAcrossLabels2Ax(ax, tests, custom_test=custom_test,
                                  columnLables=columnLables, graphLabel=graphLabel, pairs=pairs,
                                  test=test, comparisons_correction=comparisons_correction)
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    else:
        plt.show()

def plotComparisonAcrossLabels_ranksum(data_dict, columnLables, graphLabel, y_axis_label='Value', h_line=None, save_path=None, dpi=300):
    """
    Generates a simplified box plot comparison across groups.
    Uses Wilcoxon Rank-Sum test for significance and mimics the structure 
    of common plotting utilities while remaining concise.

    Args:
        data_dict (dict): Dictionary where keys are group labels (e.g., 'HC') 
                          and values are the 1D data arrays (parcel averages).
        columnLables (list of str): The group labels, specifying plot order (e.g., ['HC', 'MCI', 'AD']).
        graphLabel (str): The main title of the plot.
        y_axis_label (str): The label for the Y-axis. Defaults to 'Value'.
        h_line (float, optional): Value to plot a horizontal line at. Defaults to None.
        save_path (str, optional): Full path (including filename) to save the plot.
        dpi (int): Dots per inch for saved image quality.
    """
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.set_context('notebook', font_scale=1.2)
    
    data = pd.concat([
        pd.DataFrame({'value': data_dict[l], 'cond': l}) 
        for l in columnLables if l in data_dict
    ], ignore_index=True)

    labels = [l for l in columnLables if l in data_dict]
    default_color_map = {'HC': '#1f77b4', 'MCI': '#ff7f0e', 'AD': '#2ca02c'}
    tab10 = sns.color_palette('tab10')
    palette_list = [default_color_map.get(label, tab10[i % len(tab10)]) for i, label in enumerate(columnLables)]
    palette = dict(zip(columnLables, palette_list))
    sns.boxplot(y="value", x="cond", data=data, ax=ax, order=labels, palette=palette, linewidth=1)
    sns.swarmplot(y="value", x="cond", data=data, ax=ax, order=labels, size=3, color=".25")
    
    pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i+1, len(labels))]
    
    max_y = data['value'].max()
    min_y = data['value'].min()
    y_range = max_y - min_y
    y_increment = y_range * 0.05 if y_range > 0 else 0.1
    yposition = max_y + (y_increment * 1.5)
    
    # (Wilcoxon Rank-Sum)
    for pair in pairs:
        group1 = data[data['cond'] == pair[0]]['value']
        group2 = data[data['cond'] == pair[1]]['value']
        
        if not group1.empty and not group2.empty:
            p_value = stats.ranksums(group1, group2).pvalue
            
            sig = '***' if p_value < 0.001 else \
                  '**' if p_value < 0.01 else \
                  '*' if p_value < 0.05 else \
                  'n.s.'
            
            idx1, idx2 = labels.index(pair[0]), labels.index(pair[1])
            ax.plot([idx1, idx2], [yposition, yposition], color='black', lw=1)
            ax.text((idx1 + idx2) / 2, yposition + (y_increment * 0.2), sig,
                    ha='center', va='bottom', color='black', fontsize=10)
            yposition += y_increment # Increment y position for the next line

    ax.set_ylim(min_y * 0.95, yposition + y_increment) 
    ax.set_xlabel('')
    ax.set_ylabel(y_axis_label)
    ax.set_title(graphLabel) # Renamed to graphLabel
    ax.set_xticklabels(labels) # Set the labels for the x-axis
    
    if h_line is not None:
        ax.axhline(h_line, color='red', linestyle='--', linewidth=1, zorder=0, alpha=0.7)
    
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.show()
    plt.show()
    plt.close(fig) 

def parcel_comparison_rsn(df, measure, rsn_name, model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot):
    # --- RSN Index Selection (Core Logic) ---
    if rsn_name == 'All':
        # Get all parcel indices
        rsn_indices = list(range(NPARCELLS))
        figure_name_rsn = f'{measure} Parcels'
    else:
        # Get RSN-specific parcel indices
        rsn_parcels = df['parcel_RSNs'].iloc[0]
        rsn_indices = [i for i, rsn in enumerate(rsn_parcels) if rsn == rsn_name]
        figure_name_rsn = f'{measure} {rsn_name} Parcels'
    # ----------------------------------------

    # 1. Determine the figure name based on measure type
    if measure in ['Tau', 'Amyloid', 'ABeta']:
        figure_name = f'{figure_name_rsn} N{NPARCELLS}'
    elif measure in ['I_norm2', 'X_norm2']:
        if model_type == 'modelfree':
            figure_name = f'{figure_name_rsn} Modelfree N{NPARCELLS}'
        else:
            figure_name = f'{figure_name_rsn} Modelbased N{NPARCELLS} sig{fit_sigma} a{fit_a}'

    group_labels = df['group'].dropna().unique()

    # Calculate the mean measure across subjects for the selected parcels
    group_measure = {}
    for group in group_labels:
        group_df = df[df['group'] == group]
        # Stack all measure arrays for the group
        stacked_measures = np.stack(group_df[measure].values)
        # Select only the target parcels (RSN or All)
        rsn_measures = stacked_measures[:, rsn_indices]
        # Calculate mean across subjects (axis=0)
        group_measure[group] = np.nanmean(rsn_measures, axis=0)
    
    # Assuming plotComparisonAcrossLabels2 is defined elsewhere
    plotComparisonAcrossLabels2(
        group_measure,
        custom_test=statannotations_permutation.stat_permutation_test,
        columnLables=group_labels, # Note: this parameter might need adjustment in your original code
        graphLabel=figure_name,
        save_path=os.path.join(save_path_plot, figure_name + '.png')
    )

    return group_measure

def subject_comparison_rsn(df, measure, rsn_name, model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot):
    # --- RSN Index Selection (Core Logic) ---
    if rsn_name == 'All':
        # Get all parcel indices
        rsn_indices = list(range(NPARCELLS))
        figure_name_rsn = f'{measure} Subjects'
        y_label_suffix = ''
    else:
        # Get RSN-specific parcel indices
        rsn_parcels = df['parcel_RSNs'].iloc[0]
        rsn_indices = [i for i, rsn in enumerate(rsn_parcels) if rsn == rsn_name]
        figure_name_rsn = f'{measure} {rsn_name} Subjects'
        y_label_suffix = f' ({rsn_name} Avg)'
    # ----------------------------------------
    
    # 1. Determine the figure name based on measure type
    if measure in ['Tau', 'Amyloid', 'ABeta']:
        figure_name = f'{figure_name_rsn} N{NPARCELLS}'
    elif measure in ['I_norm2', 'X_norm2']:
        if model_type == 'modelfree':
            figure_name = f'{figure_name_rsn} Modelfree N{NPARCELLS}'
        else:
            figure_name = f'{figure_name_rsn} Modelbased N{NPARCELLS} sig{fit_sigma} a{fit_a}'

    group_labels = df['group'].dropna().unique()

    # Calculate the mean measure across RSN parcels for each subject
    group_measure = {}
    for group in group_labels:
        group_df = df[df['group'] == group]
        # Stack all measure arrays for the group
        stacked_measures = np.stack(group_df[measure].values)
        # Select only the target parcels (RSN or All)
        rsn_measures = stacked_measures[:, rsn_indices]
        # Calculate mean across parcels (axis=1) for each subject
        group_measure[group] = np.nanmean(rsn_measures, axis=1)

    # Assuming plotComparisonAcrossLabels_ranksum is defined elsewhere
    plotComparisonAcrossLabels_ranksum(
        group_measure,
        columnLables=group_labels,
        graphLabel=figure_name,
        y_axis_label=f'Subject {measure}{y_label_suffix}',
        save_path=os.path.join(save_path_plot, figure_name + '.png')
    )
    
    return group_measure


def parcel_comparison(df, measure, model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot):
    return parcel_comparison_rsn(
        df=df,
        measure=measure,
        rsn_name='All', # Pass 'All' to select all parcels
        NPARCELLS=NPARCELLS,
        fit_sigma=fit_sigma,
        fit_a=fit_a,
        save_path_plot=save_path_plot,
        model_type=model_type
    )

def subject_comparison(df, measure, model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot):
    return subject_comparison_rsn(
        df=df,
        measure=measure,
        rsn_name='All', # Pass 'All' to select all parcels
        NPARCELLS=NPARCELLS,
        fit_sigma=fit_sigma,
        fit_a=fit_a,
        save_path_plot=save_path_plot,
        model_type=model_type
    )


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------EOF
