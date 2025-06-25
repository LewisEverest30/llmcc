import pickle


'''
原始数据集文件：
    - 每条数据：60个特征序列，分别是20个rtt序列、20个loss序列、20个吞吐量序列
处理后的数据集文件：
    - 每条数据：某一时刻的3个特征，分别是rtt、loss、吞吐量（吞吐量进行了缩放）
'''


def process_dataset(old_dataset_path, new_dataset_path):
    with open(old_dataset_path, 'rb') as f:
        raw_data = pickle.load(f)  # len1 * 61
    
    processed_data = []
    for j in range(len(raw_data)):
        raw_batch_data = raw_data[j]    # [rtt1, rtt2, ..., rtt20, loss1, loss2, ..., loss20, tp1, tp2, ..., tp20, tp21]
        for i in range(20):
            processed_data.append([raw_batch_data[i], raw_batch_data[i+20], raw_batch_data[i+40]/1000])

    with open(new_dataset_path, 'wb') as f:
        pickle.dump(processed_data, f)


if __name__ == '__main__':
    process_dataset(
        old_dataset_path='./data/data_nopac.pkl',
        new_dataset_path='./data/data_nopac_processed.pkl'
    )