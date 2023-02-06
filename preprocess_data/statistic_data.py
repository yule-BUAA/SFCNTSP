import json


def statistic_data(data_path):
    with open(data_path, 'r') as file:
        users_baskets_json = json.load(file)

    users_baskets = users_baskets_json['test']

    items_set, set_count, item_count = set(), 0, 0
    for user_baskets in users_baskets:
        set_count += len(user_baskets)
        for basket in user_baskets:
            item_count += len(basket['items_id'])
            items_set = items_set.union(basket['items_id'])

    # statistics of the dataset
    print(f'number of users: {len(users_baskets)}')
    print(f'number of items: {len(items_set)}')
    print(f'number of sets: {set_count}')
    print(f'number of items per set: {item_count / set_count}')
    print(f'number of sets per user: {set_count / len(users_baskets)}')


if __name__ == "__main__":
    for dataset_name in ['JingDong', 'DC', 'TaoBao', 'TMS']:
        print(f'Statistics on {dataset_name}')
        statistic_data(data_path=f'../dataset/{dataset_name}/{dataset_name}.json')
