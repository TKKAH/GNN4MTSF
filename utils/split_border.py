def split_data_border(df_len, seq_len, split_type, train_ratio, test_ratio):
    if split_type == 'time':
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        return border1s, border2s
    elif split_type == 'amount':
        return split_data_by_ratio(df_len, seq_len, train_ratio, test_ratio)


def split_data_by_ratio(df_len, seq_len, train_ratio, test_ratio):
    num_train = int(df_len * train_ratio)
    num_test = int(df_len * test_ratio)
    num_vali = df_len - num_train - num_test
    border1s = [0, num_train - seq_len, df_len - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, df_len]
    return border1s, border2s
