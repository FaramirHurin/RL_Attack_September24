def process_positive(x):
    return max(x, 0)

def process_coordinates(x):
    return int(max(0, min(200, x)))

def process_hour(x):
    x = max(x, 0)
    x = min(x, 24)
    return int(x)

def process_remote(x):
    if x < 0.5:
        return 0
    else:
        return 1


def process_data(df):
    df['amount'] = df['amount'].apply(process_positive).round(2)
    df['payee_x'] = df['payee_x'].apply(process_coordinates)
    df['payee_y'] = df['payee_y'].apply(process_coordinates)
    df['hour'] = df['hour'].apply(process_hour)
    #df['is_online'] = df['is_online'].apply(process_remote)
    return df