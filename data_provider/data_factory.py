from torch.utils.data import DataLoader

from data_provider.data_loader import MTS_Dataset, MSTS_Dataset

data_dict = {
    'ETTh1': MTS_Dataset,
    'ETTh2': MTS_Dataset,
    'ETTm1': MTS_Dataset,
    'ETTm2': MTS_Dataset,
    'HKda': MSTS_Dataset
}


def data_provider(args, flag, logger):
    timeenc = 0 if args.embed != 'timeF' else 1
    DataSet = data_dict[args.data]
    if flag == 'test':
        shuffle_flag = False
        batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        batch_size = args.batch_size  # bsz for train and valid
    data_set = DataSet(
        root_path=args.root_path,
        data_path=args.data_path,
        split_type=args.split_type,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        freq=args.freq,
        timeenc=timeenc,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=args.scale,
        scale_type=args.scale_type,
        logger=logger,
        scale_column_wise=args.scale_column_wise

    )
    logger.info(flag+": "+str(len(data_set)))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=True)
    return data_set, data_loader
