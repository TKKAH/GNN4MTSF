def log_args(args, logger):
    logger.info("\033[1m" + "Basic Config" + "\033[0m")
    logger.info(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    logger.info(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')

    logger.info("\033[1m" + "Data Loader" + "\033[0m")
    logger.info(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    logger.info(f'  {"Data Path:":<20}{args.data_path:<20}')
    logger.info(f'  {"Checkpoints:":<20}{args.checkpoints:<20}{"Freq:":<20}{args.freq:<20}')
    logger.info(f'  {"Split Type:":<20}{args.split_type:<20}')
    logger.info(f'  {"Train Ratio:":<20}{args.train_ratio:<20}{"Test Ratio:":<20}{args.test_ratio:<20}')
    logger.info(f'  {"Scale:":<20}{args.scale:<20}{"Scale Type:":<20}{args.scale_type:<20}')
    logger.info(f'  {"Scale Column Wise:":<20}{args.scale_column_wise:<20}')
    logger.info(f'  {"Predefined Graph:":<20}{args.predefined_graph:<20}{"Embed:":<20}{args.embed:<20}')
    logger.info(f'  {"Graph Path:":<20}{args.graph_path:<20}')
    if args.task_name in ['long_term_forecast']:
        logger.info("\033[1m" + "Forecasting Task" + "\033[0m")
        logger.info(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Pred Len:":<20}{args.pred_len:<20}')
        logger.info(f'  {"Input Dim:":<20}{args.input_dim:<20}{"Output dim:":<20}{args.output_dim:<20}')
        logger.info(f'  {"Num_nodes:":<20}{args.num_nodes:<20}{"Dropout:":<20}{args.dropout:<20}')
        logger.info(f'  {"Inverse:":<20}{args.inverse:<20}')

    logger.info("\033[1m" + "Model Parameters" + "\033[0m")
    args_dict = vars(args)
    model_args = {key: value for key, value in args_dict.items() if key.startswith(args.model)}
    for key, value in model_args.items():
        logger.info(f'  {key + ":":<20}{value:<20}')
    logger.info("\033[1m" + "Run Parameters" + "\033[0m")
    logger.info(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    logger.info(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    logger.info(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    logger.info(f'  {"Loss With Regularization:":<20}{args.loss_with_regularization:<20}{"Loss:":<20}{args.loss:<20}')
    logger.info(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    
    logger.info("\033[1m" + "GPU" + "\033[0m")
    logger.info(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    logger.info(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')

def get_model_args(args):
    args_dict = vars(args)
    model_args = {key: value for key, value in args_dict.items() if key.startswith(args.model)}
    model_args_str = ''
    for key, value in model_args.items():
        model_args_str += f'{key}_{value}_'
    model_args_str = model_args_str.rstrip('_')  # 去掉最后一个逗号和空格
    return model_args_str