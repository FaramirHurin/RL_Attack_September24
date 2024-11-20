import pandas as pd
import requests
import os


def generate_kaggle_dataset():
    url = "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"
    output_file = "data.zip"

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(output_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("Download completed successfully.")
    else:
        print("Failed to download file:", response.status_code)


    df = pd.read_csv('data.zip')
    df = df.drop('Time', axis =1)
    df.rename(columns={'Class': 'label'}, inplace=True)
    df.head()

    parent_dir = os.path.abspath('.')
    path_to_save = os.path.join(parent_dir, 'Dataset', 'Kaggle_Dataset', 'creditcard.csv')
    df.to_csv(path_to_save, index=False)
