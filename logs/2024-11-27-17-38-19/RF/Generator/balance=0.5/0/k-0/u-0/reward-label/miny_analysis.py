import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def plot_miny_analysis(data, title, save_path):
    #data = data.drop(columns=['Unnamed: 0.1'])
    data.rolling(window=10).mean().plot()
    plt.title(title)
    plt.show()


data = pd.read_csv('file.csv')
plot_miny_analysis(data, 'Average K-U', 'save_path')

