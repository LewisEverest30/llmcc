import pickle
import os

RAW_DATA_BASE_PATH = './data/raw/{}/'
# CC = ['gcc', 'bbr', 'pcc']
CC = ['pcc']
PROCESSED_DATA_BASE_PATH = './data/'

def process_dataset():
    for cc in CC:
        all_data = []
        old_dataset_path = RAW_DATA_BASE_PATH.format(cc)
        for file in os.listdir(old_dataset_path):
            print(file)
            with open(old_dataset_path + file, 'r') as f:
                for line in f:
                    net_perfom = line.strip().split()
                    
                    rtt = float(net_perfom[2])
                    loss = float(net_perfom[3])
                    throughput = float(net_perfom[4])/1000
                    if throughput < 0:
                        print(f"throughput: {throughput}")
                        all_data.append([rtt, loss, 0])
                    else:
                        all_data.append([rtt, loss, throughput])

        with open(PROCESSED_DATA_BASE_PATH + cc + '.pkl', 'wb') as f:
            pickle.dump(all_data, f)

if __name__ == '__main__':
    process_dataset()