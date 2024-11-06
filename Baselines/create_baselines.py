from typing import Literal
import os
import pandas as pd
from baseline_classes import BaselineAgent

def get_dataset_list(dataset_type: Literal["Generator", "Kaggle", "SkLearn"]) -> object:
    current_dir = os.getcwd()
    dictionary = {'Kaggle':'Kaggle_Dataset', 'Generator': 'Generator_Dataset', 'SkLearn': 'SkLearn_Dataset'}
    csv_folder_path = os.path.join(current_dir, '..', 'Dataset', dictionary[dataset_type])

    csv_dict = {}
    # Loop through all files in the folder
    for file_name in os.listdir(csv_folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(csv_folder_path, file_name)
            df = pd.read_csv(file_path).iloc[:, 1:]
            csv_dict[file_name[:-4]] = df
    return csv_dict



#TODO
def get_tr(df, ratio):
    df_negative = df[df['label']==0]
    if ratio > 1:
        number = ratio
    else:
        number = ratio * df_negative.shape[0]
    return df_negative.sample(number, axis = 0)




RATIOS = [100, 2000, 0.1]
dataset_type = "SkLearn"
baselines = ['uniform', 'multivariate', 'univariate', 'mixture']

#TODO ADD CONTROLLABLE FEATURES

data_dictionary = get_dataset_list(dataset_type)
for key in data_dictionary.keys():
    df = data_dictionary[key]
    for ratio in RATIOS:
        base_dir = os.path.join(os.getcwd(), dataset_type, key, str(ratio))
        os.makedirs(base_dir, exist_ok=True)
        tr = get_tr(df, ratio)
        makedir(ratio)
        for baseline_type in baselines:
            baseline = BaselineAgent(k=controllable_features, test_x=tr, generation_method=baseline_type, quantile=0.025)
            
            store_baseline #TODO

