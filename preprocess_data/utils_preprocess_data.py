import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
import datetime


class Set(object):
    """
    Class for a single Set
    """

    def __init__(self, user_id: int or str, items_id: list, set_time: datetime.date or int):
        # id for user and items
        self.user_id = user_id  # user id
        self.items_id = items_id  # items
        # set_time
        self.set_time = set_time  # set appearing time, datetime.date or int
        self.set_semantic_time_feat = list()  # set semantic time feature, list
        # set_delta_t
        self.set_delta_t = None  # last occurrence time of user

    def to_json(self) -> dict:
        """
        convert Set object to json format
        :return:
        """
        if isinstance(self.set_time, datetime.date):
            time_str = self.set_time.strftime("%Y-%m-%d")
        else:
            time_str = self.set_time
        return {
            'user_id': self.user_id,
            'items_id': self.items_id,
            'set_time': time_str,
            'set_semantic_time_feat': self.set_semantic_time_feat,
            'set_delta_t': self.set_delta_t
        }


def get_frequent_items(transaction_df: pd.DataFrame, frequency_rate: float, key: str = 'product_id'):
    """
    get frequent items based on frequency_rate

    :param transaction_df: pd.DataFrame
    :param frequency_rate
    :param key
    :return:
        new_df: pd.DataFrame['product_id',...]
    """
    value_counts = transaction_df[key].value_counts()
    total_number = len(transaction_df)
    sum_number = 0
    item_list = []
    for index in tqdm(value_counts.index):
        if sum_number / total_number >= frequency_rate:
            break
        sum_number += value_counts[index]
        item_list.append(index)

    new_df = transaction_df[transaction_df[key].isin(item_list)]
    return new_df


def save_as_json(data: dict or list, path: str):
    """
    save data as json file with path

    :param data:
    :param path:
    :return:
    """
    with open(path, "w") as file:
        file.write(json.dumps(data))
        file.close()
        print(f'{path} writes successfully.')


def reindex_items_users(users_baskets: list, items_map_dic_path: str, users_map_dic_path: str):
    """
    reindex item id and user id in baskets

    :param users_baskets: list, [[user_1's Basket_1, user_1's Basket_2, ...], [user_2's Basket_1, user_2's Basket_2, ...], ...]
    :param items_map_dic_path:
    :param users_map_dic_path:
    :return:
    """
    # reindex item id
    items_list = []
    for user_baskets in users_baskets:
        for basket in user_baskets:
            items_list.extend(basket.items_id)

    unique_item_id_list = list(set(items_list))
    unique_item_id_list.sort()

    # generate item reindex mapping
    item_id_map_dic = defaultdict(int)
    for index, value in enumerate(unique_item_id_list):
        item_id_map_dic[value] = index
    save_as_json(item_id_map_dic, items_map_dic_path)

    # reindex item
    for user_baskets in users_baskets:
        for basket in user_baskets:
            for index, item in enumerate(basket.items_id):
                basket.items_id[index] = item_id_map_dic[item]

    # reindex user id
    user_id_map_dic = defaultdict(int)
    for index, user_baskets in enumerate(users_baskets):
        # [Basket,...]
        user_id_map_dic[user_baskets[0].user_id] = index
        for basket in user_baskets:
            basket.user_id = index
    save_as_json(user_id_map_dic, users_map_dic_path)


def get_set_time_information(user_baskets: list):
    """
    first get set semantic time feature (e.g., day)
    then get set relative time
    basket.set_time can be datetime.date or int

    :param user_baskets: List[Basket,...], time ascending
    :return:
    """
    # get semantic time feature
    # (year, month, day, weekday, isworkday) for datetime.date and a single integer for int
    for basket in user_baskets:
        if isinstance(basket.set_time, datetime.date):
            weekday = basket.set_time.weekday()
            isworkday = 0 if 0 <= weekday <= 4 else 1
            basket.set_semantic_time_feat = [basket.set_time.year, basket.set_time.month, basket.set_time.day,
                                             weekday, isworkday]
        elif isinstance(basket.set_time, int):
            basket.set_semantic_time_feat = [basket.set_time]
        else:
            raise ValueError("Time information is not datetime.date or int")

    # transform to relative time according to last basket of the user
    last_time = None
    for basket in user_baskets:
        if last_time is None:  # first basket of the user
            basket.set_delta_t = 0
        else:
            if isinstance(basket.set_time, datetime.date):
                basket.set_delta_t = (basket.set_time - last_time).days
            elif isinstance(basket.set_time, int):
                basket.set_delta_t = basket.set_time - last_time
            else:
                raise ValueError("Time information is not datetime.date or int")
            assert basket.set_delta_t > 0
        last_time = basket.set_time
