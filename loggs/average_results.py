import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
import os
import pandas as pd
import numpy as np

def organize_files_in_dict(parent_folder):
    file_dict = defaultdict(list)  # Use defaultdict to group files under the same keys
    for root, _, files in os.walk(parent_folder):
        # Relative path from the parent folder
        relative_path = os.path.relpath(root, parent_folder)
        # Split the relative path into components
        path_parts = tuple(relative_path.split(os.sep))

        # Add files to the dictionary under the key created by path_parts
        for file in files:
            if file.endswith(".csv"):
                # Build the full path to the file
                file_path = os.path.join(root, file)
                # Read the CSV and append it to the dictionary
                file_dict[path_parts].append(pd.read_csv(file_path))

    return dict(file_dict)  # Convert defaultdict to regular dict before returning
# Convert defaultdict to a regular dict before returning

def group_dataframes_by_dict(files_dict):
    grouped_files = {}
    for key in files_dict.keys():
        if len(key) > 1:
            # Exclude the first part (A) to create the new key
            key_without_a = tuple(key[1:])
            # Add files to the dictionary under the key created by excluding A
            if key_without_a not in grouped_files:
                grouped_files[key_without_a] = files_dict[key]
            else:
                grouped_files[key_without_a].append(files_dict[key])
    return grouped_files  #

def get_mean_var(dataframes:list):
    # Convert the DataFrames to a 3D NumPy array
    data_list = []
    for df in dataframes:
        if type(df) == list and len(df) == 1:
            df = df[0]
        data_list.append(df)
    data_array = np.stack(data_list, axis=0)

    # Compute the mean and variance along the first axis (across DataFrames)
    mean_array = np.mean(data_array, axis=0)
    var_array = np.var(data_array, axis=0)

    # Convert the results back to DataFrames with the same index and columns
    mean_df = pd.DataFrame(mean_array, index=dataframes[0].index, columns=dataframes[0].columns)
    var_df = pd.DataFrame(var_array, index=dataframes[0].index, columns=dataframes[0].columns)

    return mean_df, var_df, len(dataframes)

def parse_and_create_folders(key, output_folder):
    # Ensure the key is a tuple
    if not isinstance(key, tuple):
        raise ValueError("Key must be a tuple")

    # Build the folder path from all elements of the key except the last one
    folder_path = os.path.join(output_folder, *key[:-1])

    # Ensure the directory structure exists
    os.makedirs(folder_path, exist_ok=True)

    return folder_path


main_folder_path = "fourth-ultimate-run"
out_path = "averaged_results4"

files_dict = organize_files_in_dict(main_folder_path)
grouped_dataframes = group_dataframes_by_dict(files_dict)

for key in grouped_dataframes.keys():
    folder = parse_and_create_folders(key, out_path)
    mean, var, count = get_mean_var(grouped_dataframes[key])
    mean.to_csv(f"{folder}/mean.csv", index=False)
    var.to_csv(f"{folder}/var.csv", index=False)
    print(count)
    DEBUG = True



