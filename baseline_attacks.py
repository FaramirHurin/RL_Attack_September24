import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import copy
from sklearn.preprocessing import MinMaxScaler
from Baselines.attack_generation import Attack_Generation
from sklearn.ensemble import IsolationForest
from Baselines.preprocess_data import process_data

latent_dim = 10
hidden_dim = 120
batch_size = 8
num_epochs= 4000
supervised = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_Data(transactions):
    q_low = transactions["amount"].quantile(0.05)
    q_hi  = transactions["amount"].quantile(0.94)
    payees_transactions = pd.merge(payees, transactions, left_on="payee_id", right_on="payee_id")

    payees_transactions = payees_transactions[(transactions["amount"] < q_hi)
                                              & (payees_transactions["amount"] > q_low)]
    gan_transactions = copy.deepcopy(payees_transactions[['remote', 'amount', 'payee_x', 'payee_y', 'hour']])
    print(gan_transactions.describe())
    y = payees_transactions["fraud"].values

    # Normalize the data
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform\
        (gan_transactions[[ 'amount', 'payee_x', 'payee_y', 'hour']].values)
    # Add the remote column
    normalized_values = torch.tensor(normalized_values).to(torch.float)
    remote = gan_transactions["remote"].values
    remote = torch.tensor(remote).to(torch.float)
    # Add the remote column to the normalized values
    normalized_values = torch.cat((remote.unsqueeze(1), normalized_values), dim=1)

    real_data = torch.tensor(normalized_values).to(torch.float)

    return real_data, scaler, y

banksys = pickle.load(open("cache/banksys.pkl", "rb"))
payers = pd.read_csv("cache/payers-10000.csv")
payees = pd.read_csv("cache/payees-10000.csv")
transactions = pd.read_csv("cache/transactions-10000-50-2023-01-01.csv")

# Process data
real_data, scaler, y = prepare_Data(transactions)

# Load the VAE model
atk_generator = Attack_Generation(device=device, criterion = nn.MSELoss(),
                              latent_dim=latent_dim, hidden_dim=hidden_dim, training_data=real_data,
                                  lr=0.0005, trees=20, supervised=supervised, y=y)
atk_generator.train( batch_size=batch_size, num_epochs=num_epochs)

# Generate synthetic data
gen_data = atk_generator._generate_batch(20000)
# gen_data has shape (N, 1 + 4) = [remote | scaled [amount, payee_x, payee_y, hour]]
inv = scaler.inverse_transform(gen_data[:, 1:])   # (N,4) in original scale
remote = gen_data[:, 0].reshape(-1, 1)            # (N,1)
full = np.hstack((remote, inv))                   # (N,5)

gen_df = pd.DataFrame(
    full,
    columns=['remote','amount','payee_x','payee_y','hour']
)
gen_df = process_data(gen_df)


print(gen_df)
print(gen_df.describe())
