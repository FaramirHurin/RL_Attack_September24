import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

classifiers = ['RF', 'DNN']
datasets = ['Kaggle', 'SkLearn'] #'Kaggle', 'SkLearn', 'Generator'] 'Kaggle',
sklearn_datasets = ['n_features=16_n_clusters=1_class_sep=8_balance=0.5',
                    'n_features=64_n_clusters=16_class_sep=8_balance=0.5', 'n_features=64_n_clusters=16_class_sep=1_balance=0.5']
SkLearn_counters = [0, 1, 2]
balance = 'balance=0.5'
times = [300,  500,  1000, 4000]
parent_folder = 'averaged_results4'#averaged_results4  averaged_results   4 for Kaggle and SKlearn
out_folder = 'NewImages'


def flatten_dict(d):
    flat = {}
    for k1, v1 in d.items():
        for k2, v2 in v1.items():
            flat[k1 + '-' + k2] = v2
    return flat

class DataFramesComputer:
    def __init__(self, classifier:str, dataset:str, balance:str, times:list, parentfolder:str,
                 SKLEAN_COUNTER:int):
        self.classifier = classifier
        self.dataset = dataset
        self.balance = balance
        self.times = times
        self.directory = self._get_directory(parentfolder, classifier, dataset, balance, SKLEAN_COUNTER)
        self.k_types, self.u_types = self._get_K_U(SKLEAN_COUNTER)
        self._get_df_dict_flat()

    def _get_K_U(self, SKLEAN_COUNTER):
        if self.dataset == 'Generator':
            k_types = ['k-0', 'k-Term', 'k-Cust', 'k-Cust_Term']
            u_types = ['u-0', 'u-Term', 'u-Cust', 'u-Cust_Term']
        elif self.dataset == 'Kaggle':
            k_types = ['k-0', 'k-7', 'k-14']
            u_types = ['u-0', 'u-7', 'u-14']
        elif self.dataset == 'SkLearn' and SKLEAN_COUNTER == 0:
            k_types = ['k-0', 'k-4', 'k-8']
            u_types = ['u-0', 'u-4', 'u-8']
        elif self.dataset == 'SkLearn' and SKLEAN_COUNTER != 0:
            k_types = ['k-0', 'k-16', 'k-32']
            u_types = ['u-0', 'u-16', 'u-32']
        return k_types, u_types

    def _get_directory(self, parentfolder, classifier, dataset, balance, SKLEAN_COUNTER):
        if dataset == 'SkLearn':
            directory = os.path.join(parentfolder, classifier, dataset, sklearn_datasets[SKLEAN_COUNTER])
        else:
            directory = os.path.join(parentfolder,  classifier, dataset, balance)
        return directory

    def _get_df_dict_flat(self):
        df_dict = {}
        for k_type in self.k_types:
            df_dict[k_type] = {}
            for u_type in self.u_types:
                try:
                    if self.dataset == 'SkLearn':
                        df_dict[k_type][u_type] = pd.read_csv(
                            os.path.join(self.directory, k_type, u_type, 'mean.csv'))  # 32_mean #16_mean
                    else:
                        df_dict[k_type][u_type] = pd.read_csv(
                            os.path.join(self.directory, k_type, u_type, 'mean.csv'))  # 32_mean #16_mean
                except:
                    pass
        self.flat_dict = flatten_dict(df_dict)

    def compute_df_ppo_and_max(self):
        array1 = np.zeros([len(self.flat_dict.keys()), len(self.times)])
        array2 = np.zeros([len(self.flat_dict.keys()), len(self.times)])
        df_max= pd.DataFrame((array1), index=self.flat_dict.keys(), columns=self.times)
        df_ppo = pd.DataFrame((array2), index=self.flat_dict.keys(), columns=self.times)

        if not self.dataset == 'Generator':
            self.sum_k_u = df_ppo.index.str.extract(r'k-(\d+)-u-(\d+)').astype(int).sum(axis=1)
            df_ppo['sum_k_u'] = self.sum_k_u .values
            df_max['sum_k_u'] = self.sum_k_u .values
            df_ppo = df_ppo.sort_values(by='sum_k_u').drop(columns='sum_k_u')
            df_max = df_max.sort_values(by='sum_k_u').drop(columns='sum_k_u')

        for key in self.flat_dict.keys():
            for time in self.times:
                df_ppo.loc[key, time] = self.flat_dict[key]['PPO'].iloc[:time].mean()
                df_without_ppo = self.flat_dict[key].drop('PPO', axis=1)
                df_max.loc[key, time] = max(df_without_ppo.iloc[:time, :].mean())
        return df_ppo, df_max

    def compute_baselines(self):
        baselines_df = pd.DataFrame()
        for key in self.flat_dict.keys():
            df = self.flat_dict[key].drop('PPO', axis=1)
            baselines_df.loc[:, key] = df.mean()
        baselines_df = baselines_df.T
        if not self.dataset == 'Generator':
            baselines_df['sum_k_u'] = self.sum_k_u.values
            baselines_df = baselines_df.sort_values(by='sum_k_u').drop(columns='sum_k_u')
        return baselines_df.T #.round(2)



def plot_kaggle_img(df_ppo, baseline,classifier, dataset, out_folder):
    #other_metrics = df_ppo.drop(columns=['Best Baseline'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Heatmap for other metrics (left side)
    sns.heatmap(df_ppo, annot=True, cmap='cividis',  vmin=0, vmax=1,ax=axes[0], cbar=False)
    #axes[0].set_title('Other Metrics')
    axes[0].set_xlabel('Time')
    #axes[0].set_ylabel('Features')
    axes[0].tick_params(left=False, bottom=False)

    # Heatmap for Baseline (right side)
    baseline.rename(columns={4000: 'Best Baseline'}, inplace=True)

    sns.heatmap(baseline, annot=True, cmap='cividis',  vmin=0, vmax=1, cbar=False, ax=axes[1])
    #axes[1].set_title('Baseline')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].tick_params(left=False, bottom=False)

    plt.tight_layout()

    fig.savefig(f'{out_folder}/Cumulative_Reward_{dataset}_{classifier}_.svg', format='svg')

    plt.show()


# Take SKLEARN, confront last column of each PPO with  df_max
def plot_sklearn_img(dict_ppo, dict_max, out_folder):
    diff_dict = {}
    for key in dict_ppo.keys():
        if 'SkLearn' in key:
            diff_dict[key] = pd.DataFrame()
            diff_dict[key]['FRAUD-RLA'] = dict_ppo[key].iloc[:, -1]
            diff_dict[key]['Best Baseline'] = dict_max[key]
    print(diff_dict)

    # Set common color scale (0 to 1)
    vmin, vmax = 0, 1

    # Create subplots for "RF" group
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # Plot RF heatmaps
    rf_titles = [key for key in diff_dict.keys() if key.startswith('RF')]
    for ax, title in zip(axes, rf_titles):
        df = diff_dict[title]
        sns.heatmap(df, annot=True, fmt=".3f", cmap="cividis", vmin=vmin, vmax=vmax, ax=ax)
        ax.set_title(title)
    fig.savefig(f'{out_folder}/Cumulative_Reward_SKLearn_RF.svg', format='svg')

    # Create subplots for "DNN" group
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # Plot DNN heatmaps
    dnn_titles = [key for key in diff_dict.keys() if key.startswith('DNN')]
    for ax, title in zip(axes, dnn_titles):
        df = diff_dict[title]
        sns.heatmap(df, annot=True, fmt=".3f", cmap="cividis", vmin=vmin, vmax=vmax, ax=ax)
        ax.set_title(title)
    fig.savefig(f'{out_folder}/Cumulative_Reward__SKLearn_DNN.svg', format='svg')


    # Show the plots
    plt.show()





dict_ppo = {}
dict_max = {}
dict_baselines = {}

for classifier in classifiers:
    for dataset in datasets:
        for SKLEAN_COUNTER in SkLearn_counters:
            df_computer = DataFramesComputer(classifier=classifier, dataset=dataset,
                                             balance=balance, times=times, parentfolder=parent_folder,
                                             SKLEAN_COUNTER=SKLEAN_COUNTER)

            df_ppo, df_max = df_computer.compute_df_ppo_and_max()
            df_to_store = df_ppo.copy()
            df_to_store['Best Baseline'] = df_max.iloc[:, -1]

            if dataset == 'SkLearn':
                df_to_store.round(2).to_latex(
                    buf=f'{out_folder}/{classifier}-{dataset}-{SKLEAN_COUNTER}.tex',
                    index=True,
                    float_format="%.2f",
                    caption="Your Table Caption Here",
                    label="tab:your_table_label",
                    column_format="l" + "c" * len(df_to_store.columns)  # Adjust column alignment as needed
                )
            else:
                df_to_store.round(2).to_latex(
                    buf=f'{out_folder}/{classifier}-{dataset}.tex',
                    index=True,
                    float_format="%.2f",
                    caption="Your Table Caption Here",
                    label="tab:your_table_label",
                    column_format="l" + "c" * len(df_to_store.columns)  # Adjust column alignment as needed
                )

            baselines_df = df_computer.compute_baselines()
            dict_ppo[classifier + '-' + dataset + '-' + str(SKLEAN_COUNTER)] = df_ppo
            dict_max[classifier + '-' + dataset + '-' + str(SKLEAN_COUNTER)] = df_max.iloc[:, -1]
            dict_baselines[classifier + '-' + dataset + '-' + str(SKLEAN_COUNTER)] = baselines_df


# Plotting
DEBUG = True


"""
plot_sklearn_img(dict_ppo, dict_max, out_folder)


for classifier in ['RF', 'DNN']:
    if 'Generator' in datasets:
        dataset = 'Generator'
        key = classifier + '-Generator-0'
        dataset = 'Generator'
    if 'Kaggle' in datasets:
        dataset = 'Kaggle'
        key = classifier + '-Kaggle-0'
        dataset = 'Kaggle'
    plot_kaggle_img(dict_ppo[key], dict_max[key].to_frame(),classifier, dataset, out_folder)

"""


