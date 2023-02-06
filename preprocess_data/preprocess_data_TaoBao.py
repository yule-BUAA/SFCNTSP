import pandas as pd
import random
from datetime import datetime
import os
from tqdm import tqdm

from utils_preprocess_data import Set, get_frequent_items, reindex_items_users, get_set_time_information, save_as_json

train_data_percent = 0.7
validate_data_percent = 0.1
frequency_rate = 0.8
min_baskets_length = 4
max_baskets_length = 20
max_basket_boundary = 5


def read_file(data_path: str) -> pd.DataFrame:
    """
    Read original csv files from the data folder.

    Args:
        data_path : the path of the data.

    Returns:
        Transactions in pd.DataFrame.
    """
    transaction_df = pd.read_csv(data_path, header=None)
    transaction_df.columns = ['customer_id', 'product_id', 'subclass', 'behavior', 'date_time']
    transaction_df = transaction_df[['customer_id', 'subclass', 'behavior', 'date_time']]

    # behavior consists of 'pv'(click), 'buy', 'cart' and 'fav'
    # only buy behavior
    transaction_df = transaction_df[transaction_df['behavior'] == "buy"]

    transaction_df['date_time'] = pd.to_datetime(transaction_df['date_time'], unit='s').astype(str)
    # year-month-day
    transaction_df['date_time'] = transaction_df['date_time'].map(lambda x: x.split(' ')[0])
    transaction_df = transaction_df.sort_values(by='date_time')

    return transaction_df


def generate_baskets(transaction_df: pd.DataFrame, items_map_dic_path: str, users_map_dic_path: str):
    """
    generate baskets

    :param transaction_df: pd.DataFrame['customer_id', 'subclass', 'behavior', 'date_time']
    :param items_map_dic_path:
    :param users_map_dic_path:
    :return:
        users_baskets: [[Basket,...],...]
    """
    random.seed(0)

    # [[user_1's Basket_1, user_1's Basket_2, ...], [user_2's Basket_1, user_2's Basket_2, ...], ...]
    users_baskets = []
    for user_id, user in tqdm(transaction_df.groupby(['customer_id'])):
        baskets = []  # [Basket,...]
        for day, trans in user.groupby(['date_time']):  # select by user and day
            product_index_list = list(set(trans['subclass'].tolist()))
            date_time = datetime.strptime(day, "%Y-%m-%d").date()
            basket = Set(user_id, product_index_list, set_time=date_time)
            baskets.append(basket)

        # sort by time
        baskets = sorted(baskets, key=lambda basket: basket.set_time, reverse=False)

        if len(baskets) < min_baskets_length:
            # drop too short sequence
            continue
        if len(baskets) > max_baskets_length:
            # trim over-length
            baskets = baskets[:random.randint(
                max_baskets_length - max_basket_boundary, max_baskets_length)]

        users_baskets.append(baskets)

    """
    reindex item id and user id
    """
    reindex_items_users(users_baskets, items_map_dic_path, users_map_dic_path)

    """
    print info
    """
    items_set, set_count, item_count = set(), 0, 0
    for user_baskets in users_baskets:
        set_count += len(user_baskets)
        for basket in user_baskets:
            item_count += len(basket.items_id)
            items_set = items_set.union(basket.items_id)

    # statistics of the dataset
    print(f'statistics: ')
    print(f'number of users: {len(users_baskets)}')
    print(f'number of items: {len(items_set)}')
    print(f'number of sets: {set_count}')
    print(f'number of items per set: {item_count / set_count}')
    print(f'number of sets per user: {set_count / len(users_baskets)}')
    print(f'date start from {transaction_df["date_time"].min()}, end at {transaction_df["date_time"].max()}')

    return users_baskets


def generate_data(users_baskets: list, out_path: str, task_type: str = 'transductive'):
    """
    1. separate train / validate / test set
    calculate delta_t
    get next basket info as ground truth

    :param users_baskets: input data, [[Basket,...],...]
    :param out_path: output file path
    :param task_type: task type, support 'transductive' and 'inductive'
    :return:
    """

    assert task_type in ['transductive', 'inductive'], f'wrong value for task_type {task_type}!'

    train_data_list, validate_data_list, test_data_list = [], [], []

    if task_type == 'transductive':
        for _, user_baskets in enumerate(users_baskets):
            # sort baskets by time
            user_baskets = sorted(user_baskets, key=lambda basket: basket.set_time, reverse=False)

            # get set time (semantic and relative)
            get_set_time_information(user_baskets)

            # each train data contains at least two baskets
            for index in range(2, len(user_baskets) - 1):
                train_data_list.append(user_baskets[:index])

            validate_data_list.append(user_baskets[:-1])
            test_data_list.append(user_baskets)
    else:
        random.seed(0)
        random.shuffle(users_baskets)

        train_user_idx = int(train_data_percent * len(users_baskets))
        validate_user_idx = int((train_data_percent + validate_data_percent) * len(users_baskets))

        for index, user_baskets in enumerate(users_baskets):
            # sort baskets by time
            user_baskets = sorted(user_baskets, key=lambda basket: basket.set_time, reverse=False)

            # get set time (semantic and relative)
            get_set_time_information(user_baskets)

            if index < train_user_idx:
                train_data_list.append(user_baskets)
            elif train_user_idx <= index < validate_user_idx:
                validate_data_list.append(user_baskets)
            else:
                test_data_list.append(user_baskets)

    for index, train_baskets in enumerate(train_data_list):
        train_data_list[index] = [basket.to_json() for basket in train_baskets]

    for index, validate_baskets in enumerate(validate_data_list):
        validate_data_list[index] = [basket.to_json() for basket in validate_baskets]

    for index, test_baskets in enumerate(test_data_list):
        test_data_list[index] = [basket.to_json() for basket in test_baskets]

    data_dict = {
        'train': train_data_list,
        'validate': validate_data_list,
        'test': test_data_list
    }

    save_as_json(data_dict, out_path)


if __name__ == "__main__":
    data_path = f"../original_data/TaoBao_Userbehavior/UserBehavior.csv"

    for task_type in ['transductive', 'inductive']:

        dataset_name = 'TaoBao' if task_type == 'transductive' else 'TaoBao_inductive'

        root_path = f'../dataset/{dataset_name}'
        os.makedirs(root_path, exist_ok=True)

        items_map_dic_path = f'{root_path}/items_map_dic.json'
        users_map_dic_path = f'{root_path}/users_map_dic.json'
        # path for file that stores each user's own sequence
        out_path = f'{root_path}/{dataset_name}.json'

        print('Reading files ...')
        transaction_df = read_file(data_path)

        print('Removing not frequent items ...')
        transaction_df = get_frequent_items(transaction_df, frequency_rate=frequency_rate, key='subclass')

        users_baskets = generate_baskets(transaction_df, items_map_dic_path, users_map_dic_path)

        print(f'Generating data file for {dataset_name} ...')
        generate_data(users_baskets, out_path, task_type=task_type)
