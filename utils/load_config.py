import os
import json
import torch
import itertools

abs_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(abs_path) as file:
    config = json.load(file)


def get_attribute(attribute_name: str, default_value=None):
    """
    get configs
    :param attribute_name: config key
    :param default_value: None
    :return:
    """
    try:
        return config[attribute_name]
    except KeyError:
        return default_value


config['data_path'] = f"{os.path.dirname(os.path.dirname(__file__))}/dataset/{get_attribute('dataset_name')}/{get_attribute('dataset_name')}.json"

# dataset specified settings
config.update(config[f"{get_attribute('dataset_name')}"])
config.pop('JingDong')
config.pop('DC')
config.pop('TaoBao')
config.pop('TMS')
config.pop('JingDong_inductive')
config.pop('DC_inductive')
config.pop('TaoBao_inductive')
config.pop('TMS_inductive')


def get_users_items_num_and_max_seq_length(data_path):
    with open(data_path, 'r') as file:
        data_dict = json.load(file)

    max_seq_length = -1
    # get users and items num
    user_ids_set, item_ids_set = set(), set()
    for data_type in data_dict:
        for user_sets in data_dict[data_type]:
            user_ids_set = user_ids_set.union({user_sets[0]['user_id']})
            item_ids_set = item_ids_set.union(set(itertools.chain.from_iterable([user_set['items_id'] for user_set in user_sets])))

            if len(user_sets) - 1 > max_seq_length:
                max_seq_length = len(user_sets) - 1

    num_users, num_items = len(user_ids_set), len(item_ids_set)

    return num_users, num_items, max_seq_length


config['num_users'], config['num_items'], config['max_seq_length'] = get_users_items_num_and_max_seq_length(config['data_path'])
config['device'] = f'cuda:{get_attribute("cuda")}' if torch.cuda.is_available() and get_attribute("cuda") >= 0 else 'cpu'
