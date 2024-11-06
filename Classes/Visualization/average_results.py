import os
import pandas as pd

def average_csv_files(base_folder, result_folder='logs/averaged_results'):
    files_dictionary = {}

    # Create the results folder if it doesn't exist
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Traverse the folder structure according to the specified hierarchy
    for sk_folder in os.listdir(base_folder):
        sk_folder_path = os.path.join(base_folder, sk_folder)
        if not os.path.isdir(sk_folder_path):
            continue

        for zero_folder in os.listdir(sk_folder_path):
            zero_folder_path = os.path.join(sk_folder_path, zero_folder)
            if not os.path.isdir(zero_folder_path):
                continue

            for features_folder in os.listdir(zero_folder_path):
                features_folder_path = os.path.join(zero_folder_path, features_folder)
                if not os.path.isdir(features_folder_path):
                    continue

                for dataset_folder in os.listdir(features_folder_path):
                    dataset_path = os.path.join(features_folder_path, dataset_folder)
                    if not os.path.isdir(dataset_path):
                        continue

                    for k_folder in os.listdir(dataset_path):
                        k_folder_path = os.path.join(dataset_path, k_folder)
                        if not os.path.isdir(k_folder_path):
                            continue

                        for u_folder in os.listdir(k_folder_path):
                            u_folder_path = os.path.join(k_folder_path, u_folder)
                            if not os.path.isdir(u_folder_path):
                                continue

                            for fraud_genuine_folder in os.listdir(u_folder_path):
                                fraud_genuine_path = os.path.join(u_folder_path, fraud_genuine_folder)
                                if not os.path.isdir(fraud_genuine_path):
                                    continue

                                for reward_folder in os.listdir(fraud_genuine_path):
                                    reward_folder_path = os.path.join(fraud_genuine_path, reward_folder)
                                    if not os.path.isdir(reward_folder_path):
                                        continue

                                    data_csv_path = os.path.join(reward_folder_path, 'file.csv')
                                    if os.path.exists(data_csv_path):
                                        df = pd.read_csv(data_csv_path)

                                        # Use a descriptive name for each aggregate group
                                        aggregate_group = f"{features_folder}_{k_folder}_{u_folder}_{fraud_genuine_folder}_{reward_folder}"

                                        # Add DataFrame to dictionary for averaging
                                        if aggregate_group not in files_dictionary:
                                            files_dictionary[aggregate_group] = []
                                        files_dictionary[aggregate_group].append(df)

    # Average the DataFrames in each group
    for key, dfs in files_dictionary.items():
        # Compute the average DataFrame
        average_df = sum(dfs) / len(dfs)

        # Save the average DataFrame to a CSV file
        average_df.to_csv(os.path.join(result_folder, f"{key}.csv"), index=False)

# Set your base folder path
base_folder_path = '../logs/2024-10-14-15-01-19'
result_folder_path = 'logs/averaged_results'

average_csv_files(base_folder_path, result_folder_path)


# THIS IS NOT GOING TO WORK FOR OTHER FILES! LIKE KAGGLE ETC.
# ALSO STORAGE LOGIC IS NOT GOOD