import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def average_csv_files(main_folder, out_path):
    # List all the subfolders starting with "_" in the main folder
    subfolders = [os.path.join(main_folder, f) for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

    # Dictionary to store cumulative sums and counts for averaging
    cumulative_data = {}
    file_shapes = None

    # Iterate through all subfolders
    for subfolder in subfolders:
        for root, dirs, files in os.walk(subfolder):
            if "file.csv" in files:
                # Read the CSV file
                file_path = os.path.join(root, "file.csv")
                df = pd.read_csv(file_path, header=None)

                # Check if the shapes are consistent
                if file_shapes is None:
                    file_shapes = df.shape
                elif df.shape != file_shapes:
                    raise ValueError(
                        f"Inconsistent shape in file {file_path}. Expected {file_shapes}, but got {df.shape}.")

                # Add to the cumulative_data dictionary
                relative_path = os.path.relpath(root, subfolder)
                if relative_path not in cumulative_data:
                    cumulative_data[relative_path] = {
                        "sum": np.zeros(df.shape),
                        "count": 0
                    }

                cumulative_data[relative_path]["sum"][1:, :] += df.values[1:, :].astype(float)
                cumulative_data[relative_path]["count"] += 1
                columns_name = ['PPO', 'multivariate-genuine-1k', 'multivariate-genuine-100%',
                                'univariate-genuine-1k', 'univariate-genuine-100%',
                                'uniform-genuine-1k', 'uniform-genuine-100%' ,'mixture-genuine-1k', 'mixture-genuine-100%']

    # Calculate averages and save the averaged CSVs
    for relative_path, data in cumulative_data.items():
        avg_values = data["sum"] / data["count"]
        output_folder = os.path.join(out_path, main_folder[8:], relative_path)
        os.makedirs(output_folder, exist_ok=True)
        #output_file = os.path.join(output_folder, "file.csv")
        result = pd.DataFrame(avg_values, columns=columns_name)

        window_size = 50
        df_ma = result.rolling(window=window_size).mean()

        """
        # Plot original and moving averages
        plt.figure(figsize=(12, 6))
        for col in result.columns:
            # plt.plot(result.index, result[col], label=f'{col} Original')
            plt.plot(df_ma.index, df_ma[col], label=f'{col} {window_size}-Day MA', linestyle='--')

        plt.title(main_folder+relative_path)
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.grid()
        plt.show()
        """
        result.to_csv(os.path.join(output_folder, "averaged_file.csv"), index=False)


def average_over_allDatasets(root_folder, out_path):
    #        main_folder_path = "../logs/2024-11-18-20-00-24/RF/Generator/balance=0.1"
    classifiers = [ 'RF','DNN' ] #'DNN', 'RF
    dataset_types = [ 'Generator', 'SkLearn', 'Kaggle' ] #, 'SkLearn' 'Generator', 'Kaggle'
    folder_paths = [os.path.join(root_folder, classifier, dataset_type)
                    for classifier in classifiers
                    for dataset_type in dataset_types]
    # Get subpaths one layer under each resulting folder path
    subpaths = {}
    for folder_path in folder_paths:

        # Get all subpaths one level under (if they exist)
        subpaths[folder_path] = [os.path.join(folder_path, subpath)
                                 for subpath in os.listdir(folder_path)
                                 if os.path.isdir(os.path.join(folder_path, subpath))]

    for outer_path in subpaths.values():
        for path in outer_path:
            average_csv_files(path, out_path)


