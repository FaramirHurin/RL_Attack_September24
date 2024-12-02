import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import os
import pandas as pd
import numpy as np

def average_csv_files(main_folder, out_path):
    # List all the subfolders starting with "_" in the main folder
    subfolders = [os.path.join(main_folder, f) for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

    # Dictionary to store cumulative sums, sums of squares, and counts for averaging and variance
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
                    raise ValueError(f"Inconsistent shape in file {file_path}. Expected {file_shapes}, but got {df.shape}.")

                # Add to the cumulative_data dictionary
                relative_path = os.path.relpath(root, subfolder)
                if relative_path not in cumulative_data:
                    cumulative_data[relative_path] = {"sum": np.zeros(df.shape), "count": 0}
                    cumulative_data[relative_path] = {
                        "sum": np.zeros(df.shape),
                        "sum_of_squares": np.zeros(df.shape),
                        "count": 0
                    }

                # Update the cumulative sums, sums of squares, and count
                cumulative_data[relative_path]["sum"] += df.values.astype(float)
                cumulative_data[relative_path]["sum_of_squares"] += np.square(df.values.astype(float))
                cumulative_data[relative_path]["count"] += 1
                columns_name = [
                    "PPO",
                    "multivariate-genuine-1k",
                    "multivariate-genuine-100%",
                    "univariate-genuine-1k",
                    "univariate-genuine-100%",
                    "uniform-genuine-1k",
                    "uniform-genuine-100%",
                    "mixture-genuine-1k",
                    "mixture-genuine-100%",
                ]

    # Calculate averages and variances, then save the results
    for relative_path, data in cumulative_data.items():
        N = data["count"]  # Number of files
        avg_values = data["sum"] / N
        variance_values = (data["sum_of_squares"] / N) - np.square(avg_values)

        # Create output folder
        output_folder = os.path.join(out_path, main_folder[-24:], relative_path)
        os.makedirs(output_folder, exist_ok=True)

        # Save average and variance as separate files
        avg_df = pd.DataFrame(avg_values, columns=columns_name)
        var_df = pd.DataFrame(variance_values, columns=columns_name)

        # Moving average (optional)
        window_size = 50
        avg_df_ma = avg_df.rolling(window=window_size).mean()

        # Save the dataframes to CSV files
        avg_df.to_csv(os.path.join(output_folder, "averaged_file.csv"), index=False)
        var_df.to_csv(os.path.join(output_folder, "variance_file.csv"), index=False)


def average_over_allDatasets(root_folder, out_path):
    #
    print(root_folder, out_path)
    classifiers = ["RF"]  #'DNN', 'RF
    dataset_types = ["Generator"]  # , 'SkLearn' 'Generator', 'Kaggle'
    folder_paths = [os.path.join(root_folder, classifier, dataset_type) for classifier in classifiers for dataset_type in dataset_types]
    # Get subpaths one layer under each resulting folder path
    subpaths = {}
    for folder_path in folder_paths:
        # Get all subpaths one level under (if they exist)
        subpaths[folder_path] = [
            os.path.join(folder_path, subpath) for subpath in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subpath))
        ]

    for outer_path in subpaths.values():
        for path in outer_path:
            average_csv_files(path, out_path)


if __name__ == "__main__":
    main_folder_path = "../loggs/2024-11-27-18-15-50/RF/Generator/balance=0.5"
    out_path = "averaged_results/from_web"

    average_over_allDatasets(main_folder_path, out_path)
