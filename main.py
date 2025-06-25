import argparse
import torch

from dataset import CcDataset, Stage2Dataset
from llmcc import *
from plm import get_plm
from autoencoder.ae_model import Autoencoder

def train(
        is_initial_training: bool,
        plm_model_name: str,
        plm_model_size: str,
        plm_model_path: str,
        ae_pretrained_model_path: str,
        train_dataset_path: str,
        device: str,
        lora_rank: int,
        epochs: int,  # 训练轮数
        continue_epoch: int,
        load_llmcc_model_path: str,  # 继续训练时载入的模型路径
        batch_size: int,
        lr: float,
        weight_decay: float,
):
    
    print("Lr", lr)
    print("Weight_decay", weight_decay)
    print("Epochs", epochs)

    # =================== 数据集 ===================
    train_dataset = CcDataset(train_dataset_path)

    # =================== 载入外部训练的AutoEncoder模型 ===================
    ae = Autoencoder()
    if is_initial_training: # 如果是初始训练，则载入外部训练的ae模型
        ae.load_state_dict(torch.load(ae_pretrained_model_path))
    ae = ae.to(device)

    # =================== 构建孪生cc模型 ===================
    plm, plm_embed_size = get_plm(
            plm_model_name=plm_model_name,
            plm_model_path=plm_model_path,
            plm_model_size=plm_model_size,
            lora_rank=lora_rank,
            device=device,
        )

    twin_cc_net = LlmCC(ae, plm, plm_embed_size)
    if not is_initial_training:  # 如果不是初始训练，则整体载入已训练的孪生cc模型
        LlmCC.load_model(twin_cc_net, load_llmcc_model_path, lora_rank)

    # =================== 训练 ===================
    train_LlmCC(
        model=twin_cc_net,
        train_dataset=train_dataset,
        epochs=epochs, 
        continue_epoch=continue_epoch if not is_initial_training else 0,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        lora_rank=lora_rank,
    )


def test(
        plm_model_name: str,
        plm_model_size: str,
        plm_model_path: str,
        test_dataset_path: str,
        load_llmcc_model_path: str,
        device: str,
        batch_size: int,
        lora_rank: int,
):
    # =================== 数据集 ===================
    test_dataset = CcDataset(test_dataset_path)

    # =================== 载入外部训练的AutoEncoder模型 ===================
    ae = Autoencoder()
    ae = ae.to(device)

    # =================== 构建孪生cc模型 ===================
    plm, plm_embed_size = get_plm(
            plm_model_name=plm_model_name,
            plm_model_path=plm_model_path,
            plm_model_size=plm_model_size,
            lora_rank=lora_rank,
            device=device,
        )

    twin_cc_net = LlmCC(ae, plm, plm_embed_size)
    LlmCC.load_model(twin_cc_net, load_llmcc_model_path, lora_rank)
    print("载入模型路径:", load_llmcc_model_path)

    # =================== 测试 ===================
    test_LlmCC(
        model=twin_cc_net,
        test_dataset=test_dataset,
        batch_size=batch_size,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    PLM_MODEL_NAME = 'qwen2'  # PLM模型名
    PLM_MODEL_SIZE = 'small'  # PLM模型大小
    # PLM_MODEL_SIZE = 'mid'  # PLM模型大小
    PLM_MODEL_BASE_PATH = './plm_model'  # PLM模型存放路径
    AE_MODEL_PATH = './autoencoder/checkpoint/model_500.pth'  # 外部训练的ae模型路径
    TWINCC_MODEL_BASE_PATH = './checkpoint'  # 孪生cc模型存放路径


    # =================== 训练参数 ===================
    TRAIN_DATASET_PATH = './data/pcc.pkl'  # 一阶段训练数据集路径
    LORA_RANK = 4  # lora rank
    BATCH_SIZE = 32  # batch size
    LR = 1e-4  # learning rate
    WEIGHT_DECAY = 1e-5  # weight decay
    EPOCHS = 30  # 训练轮数

    LOAD_MODEL_EPOCH = 30  # 继续训练时候载入的模型的epoch
    PARAMETERS = "batch_size_32_lr_0.0001_weight_decay_1e-05_lora_rank_4"
    
    TEST_DATASET_PATH = './data/pcc.pkl'  # 测试数据集路径

    # =================== 启动参数 ===================
    parser.add_argument('--is_initial_training', action='store_true',
                        help='是否是初始训练 (即只在外部训练了ae模型，尚未整体训练整体模型) (default: False)')
    parser.add_argument('--train', action='store_true',
                        help='是否进行训练 (default: False)')
    parser.add_argument('--test', action='store_true',
                        help='是否进行测试 (default: False)')
    args = parser.parse_args()
    IS_INITIAL_TRAINING = args.is_initial_training
    TRAIN = args.train
    TEST = args.test
    print(f'IS_INITIAL_TRAINING: {IS_INITIAL_TRAINING}')
    print(f'TRAIN: {TRAIN}')
    print(f'TEST: {TEST}')

    if TRAIN:
        train(
            is_initial_training=IS_INITIAL_TRAINING,
            plm_model_name=PLM_MODEL_NAME,
            plm_model_size=PLM_MODEL_SIZE,
            plm_model_path=f'{PLM_MODEL_BASE_PATH}/{PLM_MODEL_NAME}/{PLM_MODEL_SIZE}',
            ae_pretrained_model_path=AE_MODEL_PATH,
            load_llmcc_model_path=f'{TWINCC_MODEL_BASE_PATH}/{PARAMETERS}/{LOAD_MODEL_EPOCH}',
            train_dataset_path=TRAIN_DATASET_PATH,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lora_rank=LORA_RANK,
            epochs=EPOCHS,
            continue_epoch=LOAD_MODEL_EPOCH,
            batch_size=BATCH_SIZE,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
    if TEST:
        test(
            plm_model_name=PLM_MODEL_NAME,
            plm_model_size=PLM_MODEL_SIZE,
            plm_model_path=f'{PLM_MODEL_BASE_PATH}/{PLM_MODEL_NAME}/{PLM_MODEL_SIZE}',
            test_dataset_path=TEST_DATASET_PATH,
            load_llmcc_model_path=f'{TWINCC_MODEL_BASE_PATH}/{PARAMETERS}/{LOAD_MODEL_EPOCH}',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch_size=BATCH_SIZE,
            lora_rank=LORA_RANK,
        )