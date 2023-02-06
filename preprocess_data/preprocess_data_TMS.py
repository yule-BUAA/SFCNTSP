from tqdm import tqdm
import random
import json
import os

from utils_preprocess_data import Set, reindex_items_users, save_as_json

train_data_percent = 0.7
validate_data_percent = 0.1
min_baskets_length = 4
max_baskets_length = 20
max_basket_boundary = 5


def generate_baskets(data_dict: dict, items_map_dict_path: str, users_map_dict_path: str):
    """
    generate baskets for all the users

    :param data_dict: data dictionary {
                                       'train' : {user_1_id: [[item_1_id, item_2_id], [item_3_id, item_5_id], ...], ...},
                                       'validate' : {user_1_id: [[item_3_id, item_5_id], [item_1_id, item_4_id], ...], ...},
                                       'test' : {user_1_id: [[item_1_id], [item_2_id, item_3_id, item_7_id], ...], ...}
                                      }
    :param items_map_dict_path: save path of items id mapping dictionary
    :param users_map_dict_path: save path of users id mapping dictionary
    :return:
        users_baskets: [[Basket,...],...]
    """

    random.seed(0)

    # [[user_1's Basket_1, user_1's Basket_2, ...], [user_2's Basket_1, user_2's Basket_2, ...], ...]

    users_baskets = []
    
    for mode in ['train', 'validate', 'test']:
        for user_id, user_baskets in tqdm(data_dict[mode].items()):
            baskets = []  # [Basket,...]
            user_id = f'{mode}_{user_id}'
            sequence_length = len(user_baskets)
            for index, basket in enumerate(user_baskets):
                assert len(basket) > 0
                basket = Set(user_id, basket, set_time=index)
                basket.set_delta_t = sequence_length - 1 - index
                baskets.append(basket)

            # sort by time
            baskets = sorted(baskets, key=lambda basket: basket.set_time, reverse=False)

            if len(baskets) < min_baskets_length:
                # drop too short sequence
                continue
            elif len(baskets) > max_baskets_length:
                # trim over-length
                baskets = baskets[:random.randint(max_baskets_length - max_basket_boundary, max_baskets_length)]

            users_baskets.append(baskets)

    """
    reindex item id and user id
    """
    reindex_items_users(users_baskets, items_map_dict_path, users_map_dict_path)

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

    return users_baskets


def generate_data(users_baskets: list, out_path: str, task_type: str = 'transductive'):
    """
    1. separate train / validate / test set
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
    data_path = "../original_data/SOS_Data/tags-math-sx-seqs.json"

    for task_type in ['transductive', 'inductive']:

        dataset_name = 'TMS' if task_type == 'transductive' else 'TMS_inductive'

        root_path = f'../dataset/{dataset_name}'
        os.makedirs(root_path, exist_ok=True)

        items_map_dict_path = f'{root_path}/items_map_dic.json'
        users_map_dict_path = f'{root_path}/users_map_dic.json'
        # path for file that stores each user's own sequence
        out_path = f'{root_path}/{dataset_name}.json'

        print('Reading files ...')
        with open(data_path, 'r') as file:
            data_dict = json.load(file)

        users_baskets = generate_baskets(data_dict, items_map_dict_path, users_map_dict_path)

        print(f'Generating data file for {dataset_name} ...')
        generate_data(users_baskets, out_path, task_type=task_type)
