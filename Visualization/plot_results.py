import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
folder = os.path.join(os.getcwd(), 'averaged_results', '2024-11-18-20-00-24')
classifier_type =  'DNN' #DNN, RF
dataset_type = 'SkLearn' #SkLearn, Kaggle, Generator
balance = 'balance=0.5'
n_features = 'n_features=64'
n_clusters = '_n_clusters=16'
class_sep = '_class_sep=2_'
"""




def reorganize_results(folder, classifier_type, dataset_type, balance, n_features, n_clusters, class_sep):
    print('Folder is'+str(folder))
    ppo_result ={}
    best_other = {}
    all_baselines = {}
    if dataset_type == 'SkLearn':
        sklearn_name = n_features+n_clusters+class_sep+balance
        file_location = os.path.join(folder, classifier_type, dataset_type, sklearn_name)
    else:
        file_location = os.path.join(folder, classifier_type, dataset_type, balance)
    k_folders = [f.path for f in os.scandir(file_location) if f.is_dir()]
    for k in k_folders:
        k_name = k.split('\\')[-1].split('/')[-1]
        all_baselines[k_name] = {}
        ppo_result[k_name] = {}
        best_other[k_name] = {}
        u_folders = [f.path for f in os.scandir(k) if f.is_dir()]
        for u in u_folders:
            u_name = u.split('\\')[-1].split('/')[-1]
            ppo_result[k_name][u_name] = {}
            file = os.path.join(u, 'reward-label', 'averaged_file.csv')
            df = pd.read_csv(file)
            all_baselines[k_name][u_name] = df.drop('PPO', axis=1).mean().round(2)

            best_other[k_name][u_name] = max(df.drop('PPO', axis=1).mean().round(2))

            BEGIN_COUNT = 0
            for i in [300, 1000, 2000, 4000]:
                first_k = df.iloc[BEGIN_COUNT:i, :].mean()
                ppo_result[k_name][u_name][i]= first_k['PPO'].round(2)
    return ppo_result, best_other, all_baselines




def plot(ppo_result, best_other):
    results = {}

    for i in [300, 1000, 2000, 4000]:
        results[i] = pd.DataFrame()
        competitors = pd.DataFrame()
        for k in ppo_result.keys():
            for u in ppo_result[k].keys():
                other = best_other[k][u]
                ppo = ppo_result[k][u][i]
                print(f'K is {k}, u is {u}, tune = {i}: {ppo}, {other}')
                results[i].loc[u, k] = float(ppo - other)
                competitors.loc[u, k]  = other
        print(results[i])
        fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    for idx, i in enumerate([300,1000, 2000, 4000]):
        sns.heatmap(results[i], annot=True, fmt=".2f", cmap='coolwarm', vmin=-1,
                    vmax=1, ax=axes[idx])
        axes[idx].set_xlabel('K')
        axes[idx].set_ylabel('U')
        axes[idx].set_title(f'PPO - Other for {i}')
    plt.tight_layout()
    plt.show()

    sns.heatmap(competitors, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel = competitors.columns
    plt.ylabel = competitors.index
    axes[idx].set_title(f'Other for {i}')
    plt.show()

def plot_baselines(all_baselines):
    d = all_baselines

    # Separate the entries into two groups: "100" and "1k"
    # Separate the entries into two groups: those containing "100%" and those containing "1k"
    group_100 = [
        (u, k, series[series.index.str.contains('100%')])
        for u, subdict in d.items()
        for k, series in subdict.items()
    ]
    group_1k = [
        (u, k, series[series.index.str.contains('1k')])
        for u, subdict in d.items()
        for k, series in subdict.items()
    ]


    # Plot the "100" group
    if group_100:
        unique_100_labels = sorted(set(idx for _, _, series in group_100 for idx in series.index if "100" in idx))
        plot_group(group_100, "Plots for Index Containing '100%'", unique_100_labels)

    # Plot the "1k" group
    if group_1k:
        unique_1k_labels = sorted(set(idx for _, _, series in group_1k for idx in series.index if "1k" in idx))
        plot_group(group_1k, "Plots for Index Containing '1k'", unique_1k_labels)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_group(group, title, index_labels, figsize=(12, 8)):
    """
    Plots a group of series with shared index labels at the figure level.
    """
    num_plots = len(group)
    nrows = int(np.ceil(np.sqrt(num_plots)))
    ncols = int(np.ceil(num_plots / nrows))

    # Create figure and axes
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=False, sharey=False, constrained_layout=True)
    axs = axs.flatten() if num_plots > 1 else [axs]

    # Define a color map to distinguish bars, but apply it to the bars, not the subfigure
    colors = plt.cm.get_cmap('tab10', 10)  # We use a colormap to assign different colors to each bar

    # Collect legend labels (index labels specific to the group)
    legend_labels = []
    bar_colors = []  # Store color for each index label for the legend


    # Plot each series
    for idx, (u, k, series) in enumerate(group):
        ax = axs[idx]
        color = colors(np.arange(len(series)))  # Assign color to bars
        # Reset index and plot, labeling bars with the index value, and apply colors to the bars
        bars = series.reset_index(drop=True).plot(kind="bar", ax=ax, color=color, )  # Assign color to bars

        # Set the title and labels
        ax.set_title(f'{u}, k={k}')
        ax.set_xlabel("")  # No xlabel
        ax.set_ylabel("Success Rate")  # Adjust as needed

    # Collect the index values for the legend (only relevant index labels for this group)
    color_index = 0
    for index_value in index_labels:

        legend_labels.append(index_value)
        bar_colors.append(colors(color_index))
        color_index += 1

    # Hide unused subplots
    for ax in axs[num_plots:]:
        ax.axis('off')

    # Add index labels at the figure level
    fig.text(
        0.5, -0.05, f"Index Labels: {', '.join(index_labels)}",
        ha='center', va='center', wrap=True, fontsize=10
    )

    # Add a global title
    fig.suptitle(title, y=1.05, fontsize=16)

    # Place the legend at the figure level outside of the subplots, with more space
    fig.legend(
        handles=[plt.Line2D([0], [0], color=color, lw=4) for color in bar_colors],
        labels=legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.9),  # Move the legend higher to create space for it
        ncol=2,  # Adjust number of columns if necessary
        frameon=False,
        fontsize=10
    )

    # Use tight_layout to adjust for everything fitting within the figure area
    plt.tight_layout(rect=[0, 0, 1, 1])  # Tighten layout and leave space for title and legend

    plt.show()
