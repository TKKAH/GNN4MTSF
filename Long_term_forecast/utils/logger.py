import logging
import os
import sys
import time


def get_log_dir(args):
    run_id = '%s-%s-%s-%s/' % (
        args.model_id, args.model, args.data,
        time.strftime('%Y_%m_%d_%H_%M_%S'))
    base_dir = args.root_path
    log_dir = os.path.join(base_dir, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger
