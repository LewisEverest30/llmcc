import pickle

def combine_datasets(dataset_path_list: list, combined_dataset_path: str):

    combined_data = []
    for dataset_path in dataset_path_list:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        combined_data += data
    
    with open(combined_dataset_path, 'wb') as f:
        pickle.dump(combined_data, f)


if __name__ == '__main__':

    dataset_path_list = [
        'data/gcc.pkl',
        'data/pcc.pkl',
        'data/bbr.pkl',
    ]
    combined_dataset_path = 'data/merge_three_cc_1.6.pkl'
    combine_datasets(dataset_path_list=dataset_path_list, combined_dataset_path=combined_dataset_path)