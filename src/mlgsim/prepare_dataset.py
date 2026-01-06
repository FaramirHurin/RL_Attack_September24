import pickle as pk
import os
import numpy as np
import pandas as pd

transactions_df = None

def preprocess():

    directory = 'simulated-data-raw/'
    transactions_df = None

    for file in os.listdir(directory):
        if file.endswith('.pkl') and file.startswith('2018'):
            filepath = os.path.join(directory, file)
            with open(filepath, 'rb') as f:
                data = pk.load(f)
            if transactions_df is None:
                transactions_df = data
            else:
                transactions_df = pd.concat([transactions_df, data], ignore_index=True)


    #pk.load(open('simulated-data-raw/2018-09-22.pkl', 'rb'))

    # Rename columns
    transactions_df.rename(columns={
        'CUSTOMER_ID': 'card_id' ,
        'TERMINAL_ID': 'terminal_id',
        'TX_FRAUD': 'is_fraud',
        'TX_AMOUNT': 'amount',
        # 'TX_TIME_DAYS': 'day_index',
        'TX_DATETIME': 'timestamp',
    }, inplace=True)

    transactions_df.drop(['TX_FRAUD_SCENARIO', 'TRANSACTION_ID', 'TX_TIME_SECONDS', 'TX_TIME_DAYS'], axis=1, inplace=True)
    transactions_df['is_online'] = np.zeros(transactions_df.shape[0])

    transactions_df['timestamp'] = transactions_df['timestamp'].dt.to_pydatetime()

    with open('simulated-data-raw/customer_profiles_table.pkl', 'rb') as f:
        customer_profiles_table = pk.load(f)

    with open('simulated-data-raw/terminal_profiles_table.pkl', 'rb') as f:
        terminal_profiles_table = pk.load(f)

    customer_profiles_table['x_customer_id'] = np.round(customer_profiles_table['x_customer_id'])
    customer_profiles_table['y_customer_id'] = np.round(customer_profiles_table['y_customer_id'])

    customer_profiles_table['balance'] = (customer_profiles_table['mean_amount'] * customer_profiles_table['mean_nb_tx_per_day'] * 60).round(2)

    terminal_profiles_table['x_terminal_id'] = np.round(terminal_profiles_table['x_terminal_id'])
    terminal_profiles_table['y_terminal_id'] = np.round(terminal_profiles_table['y_terminal_id'])




    customer_profiles_table.rename(columns={
        'CUSTOMER_ID': 'id',
        'x_customer_id': 'x',
        'y_customer_id': 'y',
    }, inplace=True)

    terminal_profiles_table.rename(columns={
        'TERMINAL_ID': 'id',
        'x_terminal_id': 'x',
        'y_terminal_id': 'y',
    }, inplace=True)

    customer_profiles_table = customer_profiles_table[['id', 'x', 'y', 'balance']]
    terminal_profiles_table = terminal_profiles_table[['id', 'x', 'y']]

    # Save the processed dataframes
    transactions_df.to_csv('transactions.csv', index=False)
    customer_profiles_table.to_csv('customer_profiles.csv', index=False)
    terminal_profiles_table.to_csv('terminal_profiles.csv', index=False)




if __name__ == "__main__":
    preprocess()