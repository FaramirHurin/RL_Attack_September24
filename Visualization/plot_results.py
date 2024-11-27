import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

folder = os.path.join(os.getcwd(), 'averaged_results', '2024-11-18-20-00-24')
classifier_type =  'DNN' #DNN, RF
dataset_type = 'SkLearn' #SkLearn, Kaggle, Generator
balance = 'balance=0.5'
n_features = 'n_features=64'
n_clusters = '_n_clusters=16'
class_sep = '_class_sep=2_'


def reorganize_results(folder, classifier_type, dataset_type, balance, n_features, n_clusters, class_sep):
    ppo_result ={}
    best_other = {}
    if dataset_type == 'SkLearn':
        sklearn_name = n_features+n_clusters+class_sep+balance
        file_location = os.path.join(folder, classifier_type, dataset_type, sklearn_name)
    else:
        file_location = os.path.join(folder, classifier_type, dataset_type, balance)
    k_folders = [f.path for f in os.scandir(file_location) if f.is_dir()]
    for k in k_folders:
        k_name = k[-4:]
        ppo_result[k_name] = {}
        best_other[k_name] = {}
        u_folders = [f.path for f in os.scandir(k) if f.is_dir()]
        for u in u_folders:
            u_name = u[-4:]
            ppo_result[k_name][u_name] = {}
            best_other[k_name][u_name] = {}
            file = os.path.join(u, 'reward-label', 'averaged_file.csv')
            df = pd.read_csv(file)
            best_other[k_name][u_name] = max(df.mean().round(2))

            BEGIN_COUNT = 200
            for i in [300, 2000, df.shape[0]]:
                first_k = df.iloc[BEGIN_COUNT:i, :].mean()
                ppo_result[k_name][u_name][i]= first_k['PPO'].round(2)
    return ppo_result, best_other

def plot(ppo_result, best_other):
    sub_ppo = ppo_result[tuple(ppo_result.keys())[0]]
    subsub_ppo = sub_ppo[tuple(sub_ppo.keys())[0]]
    for i in subsub_ppo.keys():
        table_data = []
        for k_name, k_data in ppo_result.items():
            for u_name, u_data in k_data.items():
                table_data.append([k_name, u_name, u_data[i]])
        df = pd.DataFrame(table_data, columns=['k', 'u', 'PPO'])
        df_table = df.pivot(index='k', columns='u', values='PPO')
        print(df_table)

        df_best = pd.DataFrame(best_other).T[df_table.columns].reindex(df_table.index)

        """
        sns.heatmap(df_table, annot=True, mask= df_table.isnull(), fmt=".2f")
        plt.title(f'PPO for {i} steps')
        plt.show()

        sns.heatmap(df_best, annot=True, mask= df_table.isnull(), fmt=".2f")
        plt.title(f'Best Comparison for {i} steps')
        plt.show()
        """

        sns.heatmap(df_table - df_best, annot=True, mask=df_table.isnull(), fmt=".2f")
        plt.title(f'Best Comparison for {i} steps')
        plt.show()

"""
ppo_results, best_other = plot_results(folder, classifier_type, dataset_type, balance,
                           n_features, n_clusters, class_sep)
plot(ppo_results, best_other)
"""