import torch
import numpy as np
import torch.nn as nn
import warnings
import os
import shutil
from tqdm import tqdm
import json
import logging
import time
from collections import defaultdict
import sys

from utils.load_config import get_attribute, config
from utils.utils import create_optimizer, create_lr_scheduler, convert_to_gpu, set_random_seed, get_n_params
from utils.metrics import get_metric, get_all_metric
from utils.data_loader import get_data_loader
from utils.EarlyStopping import EarlyStopping
from model.SFCNTSP import SFCNTSP


def evaluate(model: nn.Module, data_loader_dic: dict, loss_func: nn.Module, logger: logging.Logger):
    """
    evaluate model
    :param model:
    :param data_loader_dic:
    :param loss_func:
    :return:
    """

    loss_dict, metric_dict = defaultdict(list), defaultdict(list)

    for mode in ["train", "validate", "test"]:

        model.eval()
        tqdm_loader = tqdm(data_loader_dic[mode], ncols=175)

        for batch, (_, batch_items_id, batch_seq_length, batch_set_size, batch_input_data, batch_truth_data) in enumerate(tqdm_loader):

            batch_input_data, batch_truth_data = convert_to_gpu(batch_input_data, batch_truth_data,
                                                                device=get_attribute('device'))

            batch_output = model(batch_seq_length=batch_seq_length, batch_items_id=batch_items_id,
                                 batch_set_size=batch_set_size, batch_input_data=batch_input_data)

            loss = loss_func(batch_output, batch_truth_data)
            loss_dict[mode].append(loss.item())
            metric_dict[mode].append(get_metric(y_true=batch_truth_data, y_pred=batch_output))
            tqdm_loader.set_description(f'{mode} batch: {batch + 1}, {mode} loss: {loss.item()}')

    train_metric = get_all_metric(metric_list=metric_dict['train'])
    val_metric = get_all_metric(metric_list=metric_dict['validate'])
    test_metric = get_all_metric(metric_list=metric_dict['test'])

    logger.info(
        f"train loss {torch.Tensor(loss_dict['train']).mean()}, valid loss: {torch.Tensor(loss_dict['validate']).mean()},"
        f"test loss: {torch.Tensor(loss_dict['test']).mean()}, \ntrain metric: {train_metric}, \nvalid metric: {val_metric}, "
        f"\ntest metric: {test_metric}")

    return train_metric, val_metric, test_metric


if __name__ == "__main__":
    """
    init dataloader, paths and logger
    """
    warnings.filterwarnings('ignore')

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    os.makedirs(f"../logs/{get_attribute('dataset_name')}/{get_attribute('model_name')}", exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"../logs/{get_attribute('dataset_name')}/{get_attribute('model_name')}/{str(time.time())}.log")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f'config -> {json.dumps(config, indent=4)}')

    set_random_seed(seed=get_attribute('seed'))

    train_data_loader = get_data_loader(data_path=get_attribute('data_path'), data_type='train',
                                        batch_size=get_attribute('batch_size'), max_seq_length=get_attribute('max_seq_length'),
                                        num_items=get_attribute('num_items'), num_workers=4)

    val_data_loader = get_data_loader(data_path=get_attribute('data_path'), data_type='validate',
                                      batch_size=get_attribute('batch_size'), max_seq_length=get_attribute('max_seq_length'),
                                      num_items=get_attribute('num_items'), num_workers=4)

    test_data_loader = get_data_loader(data_path=get_attribute('data_path'), data_type='test',
                                       batch_size=get_attribute('batch_size'), max_seq_length=get_attribute('max_seq_length'),
                                       num_items=get_attribute('num_items'), num_workers=4)

    data_loader_dic = {"train": train_data_loader, "validate": val_data_loader, "test": test_data_loader}

    model = SFCNTSP(num_items=get_attribute('num_items'), max_seq_length=get_attribute('max_seq_length'),
                    embedding_channels=get_attribute('embedding_channels'), dropout=get_attribute('dropout'),
                    bias=get_attribute('bias'), alpha=get_attribute('alpha'), beta=get_attribute('beta'))

    model = convert_to_gpu(model, device=get_attribute('device'))

    logger.info(model)
    logger.info(f'Model #Params: {get_n_params(model) * 4} B, {get_n_params(model) * 4 / 1024} KB, {get_n_params(model) * 4 / 1024 / 1024} MB.')

    optimizer = create_optimizer(model=model, optimizer_name=get_attribute('optimizer'), learning_rate=get_attribute('learning_rate'),
                                 weight_decay=get_attribute('weight_decay'))

    scheduler = create_lr_scheduler(optimizer=optimizer, learning_rate=get_attribute('learning_rate'),
                                    t_max=get_attribute('scheduler_t_max'))

    save_model_folder = f"../save_model_folder/{get_attribute('dataset_name')}/{get_attribute('model_name')}"
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)

    early_stopping = EarlyStopping(patience=get_attribute('patience'), save_model_folder=save_model_folder,
                                   save_model_name=get_attribute('model_name'), logger=logger)

    loss_func = nn.MultiLabelSoftMarginLoss(reduction="mean")

    for epoch in range(get_attribute('epochs')):

        loss_dict, metric_dict = defaultdict(list), defaultdict(list)

        for mode in ["train", "validate", "test"]:
            # training
            if mode == "train":
                model.train()
            # validate or test
            else:
                model.eval()

            tqdm_loader = tqdm(data_loader_dic[mode], ncols=150)

            for batch, (_, batch_items_id, batch_seq_length, batch_set_size, batch_input_data, batch_truth_data) in enumerate(tqdm_loader):

                batch_input_data, batch_truth_data = convert_to_gpu(batch_input_data, batch_truth_data, device=get_attribute('device'))

                with torch.set_grad_enabled(mode == 'train'):
                    batch_output = model(batch_seq_length=batch_seq_length, batch_items_id=batch_items_id,
                                         batch_set_size=batch_set_size, batch_input_data=batch_input_data)

                    loss = loss_func(batch_output, batch_truth_data)
                    if mode == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        if get_attribute('use_scheduler'):
                            scheduler.step()

                    loss_dict[mode].append(loss.item())
                    metric_dict[mode].append(get_metric(y_true=batch_truth_data, y_pred=batch_output))
                    tqdm_loader.set_description(f'{mode} epoch: {epoch + 1}, batch: {batch + 1}, {mode} loss: {loss.item()}')

        train_metric = get_all_metric(metric_list=metric_dict['train'])
        val_metric = get_all_metric(metric_list=metric_dict['validate'])
        test_metric = get_all_metric(metric_list=metric_dict['test'])

        logger.info(f"Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]['lr']}, train loss {torch.Tensor(loss_dict['train']).mean()}, valid loss: {torch.Tensor(loss_dict['validate']).mean()},"
                    f"test loss: {torch.Tensor(loss_dict['test']).mean()}, \ntrain metric: {train_metric}, \nvalid metric: {val_metric}, "
                    f"\ntest metric: {test_metric}")

        # save best model using validate data
        validate_ndcg = np.mean([val_metric[key] for key in val_metric if key.startswith(f"ndcg_")])
        early_stop = early_stopping.step([('ndcg', validate_ndcg, True)], model)

        if early_stop:
            break

    # load best model and calculate final metrics
    early_stopping.load_checkpoint(model)

    logger.info('calculating final metrics...')

    train_metric, val_metric, test_metric = evaluate(model, data_loader_dic, loss_func, logger=logger)

    save_result_folder = f"../results/{get_attribute('dataset_name')}"
    os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = f"{save_result_folder}/{get_attribute('model_name')}.json"

    with open(save_result_path, 'w') as file:
        scores_str = json.dumps({"train": train_metric, "validate": val_metric, "test": test_metric}, indent=4)
        file.write(scores_str)
        logger.info(f'result saves at {save_result_path} successfully.')

    sys.exit()
