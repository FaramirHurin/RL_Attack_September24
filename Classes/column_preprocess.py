import polars as pl
from itertools import product
import random
import math
import numpy as np
def process_generator_columns(df):
    C_columns_fixed = ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT']
    remaining_columns = [col for col in df.columns if col not in C_columns_fixed]
    customer_id_columns = [col for col in remaining_columns if col.startswith("CUSTOMER_ID")]
    terminal_columns = [col for col in remaining_columns if col.startswith("TERMINAL")]

    combinations = []

    for customer_k, customer_u, customer_c in product(
            [customer_id_columns, []], [customer_id_columns, []], [customer_id_columns, []]):

        if len(customer_k) + len(customer_u) + len(customer_c) != len(customer_id_columns):
            continue

        for terminal_k, terminal_u, terminal_c in product(
                [terminal_columns, []], [terminal_columns, []], [terminal_columns, []]):

            if len(terminal_k) + len(terminal_u) + len(terminal_c) != len(terminal_columns):
                continue

            K_columns = customer_k + terminal_k
            U_columns = customer_u + terminal_u
            C_columns = C_columns_fixed + customer_c + terminal_c

            combinations.append({
                'K_columns': K_columns,
                'U_columns': U_columns,
                'C_columns': C_columns
            })
    return combinations


def sample_columns(df): #, seed=42
    #random.seed(seed)
    all_columns = df.columns

    columns_number = len(df.columns)
    forth =  math.floor(columns_number / 4)
    u_values =  np.array(range(3))  * forth
    k_values =  np.array(range(3)) * forth
    results = []
    combinations = []
    for u in u_values:
        for k in k_values:
            if u + k <  4 * forth:
                combination = {'k_size': k, 'u_size': u}
                combinations.append(combination)

    for combination in combinations:
        shuffled_columns = random.sample(all_columns, len(all_columns))
        k_size = combination['k_size']
        u_size = combination['u_size']
        c_size = columns_number - k_size - u_size
        print('Number of controllable features is ' + str(c_size))

        K_columns = shuffled_columns[:k_size]
        U_columns = shuffled_columns[k_size:k_size + u_size]
        C_columns = shuffled_columns[k_size + u_size:]

        # Append the combination as a dictionary
        results.append({
            'K_columns': K_columns,
            'U_columns': U_columns,
            'C_columns': C_columns
        })

    return results


def get_column_combinations(dataset_type: str, df: pl.DataFrame): #, seed=42
    match dataset_type:
        case 'SkLearn':
            return sample_columns(df)
        case 'Kaggle':
            return sample_columns(df)
        case 'Generator':
            return process_generator_columns(df)
        case _:
            raise ('Not a valid dataset type')






