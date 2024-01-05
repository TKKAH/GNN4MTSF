def split_data_border(df_len, seq_len, split_type):
    border1s = None
    border2s = None
    if split_type == 'time':
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif split_type == 'amount':
        num_train = int(df_len * 0.7)
        num_test = int(df_len * 0.2)
        num_vali = df_len - num_train - num_test
        border1s = [0, num_train - seq_len, df_len - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, df_len]
    return border1s, border2s
