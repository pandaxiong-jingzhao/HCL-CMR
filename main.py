import argparse
import json
import os
import csv
import time
import math
from copy import deepcopy

import numpy as np
from dataset import get_dataset, get_handler, is_openml
from itr_models.model import DALGNN
from models.model import get_net, MLPNet
from sklearn.manifold import TSNE
from torchvision import transforms
import torch
from torch.utils.tensorboard import SummaryWriter
from query_strategies import *
from models.lenet import LeNet5
from models.resnet import ResNetClassifier
from models.vgg import VGGClassifier
from models.densenet import DenseNetClassifier
from models.vision_transformer import VisionTransformerClassifier
import matplotlib.pyplot as plt
import sys
from models.training import Training
from models.cdal_model import CDALModel
import matplotlib
import matplotlib.colors as mcolors
from scipy.io import loadmat

from itr_models.clip4cmr import CLIP4CMRModel
from models.clip4cmr_training import CLIP4CMRTraining
from query_strategies.vmfal_sampling import VMFSampling
from query_strategies.multi_modality_sampling import MultiModalSampling
from itr_models.albef4cmr import ALBEF4CMRModel
from itr_models.vlmo4cmr import VLMo4CMRModel
# ================================================================
# 修改 1：文件顶部添加导入（在其他 import 之后）
# ================================================================
from query_strategies.typiclust_sampling import TypiClustSampling
from query_strategies.probcover_sampling import ProbCoverSampling
# FullSupervised 已经通过 query_strategies/__init__.py 的 * 导入
# 但为安全起见，也显式导入:
from query_strategies.full_supervised import FullSupervised

from query_strategies.ccma_sampling import CCMASampling
from query_strategies.rlmba_sampling import RLMBASampling

# --- 新增：强制使用文件系统共享内存，彻底解决 DupFd 和 _ForkingPickler 报错 ---
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



# ===================== Hyper-parameter sweep scoring helpers =====================
def compute_al_curve_stats(
    round_metrics_csv: str,
    metric_col: str = "test_mAP",
    stable_alpha: float = 1.0,
    stable_beta: float = 1.0,
):
    """
    Compute learning-curve statistics from round_metrics.csv.

    Returns a dict with:
      - mean_all: mean metric across all rounds
      - std: standard deviation across rounds
      - max_drop: maximum drawdown (largest drop from a previous peak to a later point)
      - stable_score: mean_all - alpha*std - beta*max_drop  (higher is better: strong + stable)
      - last: last round metric
      - last2: mean of last 2 rounds metric (more robust than last)
      - n_rounds: number of valid rounds used
    """
    if not os.path.exists(round_metrics_csv):
        raise FileNotFoundError(f"round_metrics.csv not found: {round_metrics_csv}")

    rows = []
    with open(round_metrics_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if len(rows) == 0:
        raise ValueError(f"round_metrics.csv is empty: {round_metrics_csv}")

    # Ensure correct order
    rows.sort(key=lambda x: int(float(x["round"])))

    y = np.array([float(r[metric_col]) for r in rows], dtype=np.float64)

    # Drop invalid placeholders (e.g. -1)
    valid = np.isfinite(y) & (y > -0.5)
    y = y[valid]
    if len(y) == 0:
        raise ValueError(f"no valid '{metric_col}' values in {round_metrics_csv}")

    mean_all = float(y.mean())
    std_all = float(y.std())

    # Max drawdown (peak-to-trough)
    peak = float(y[0])
    max_drop = 0.0
    for v in y[1:]:
        v = float(v)
        if v > peak:
            peak = v
        else:
            max_drop = max(max_drop, peak - v)

    last = float(y[-1])
    last2 = float(y[-2:].mean()) if len(y) >= 2 else last

    stable_score = float(mean_all - stable_alpha * std_all - stable_beta * max_drop)

    return {
        "mean_all": mean_all,
        "std": std_all,
        "max_drop": float(max_drop),
        "stable_score": stable_score,
        "last": last,
        "last2": last2,
        "n_rounds": int(len(y)),
    }
# ===================== End sweep scoring helpers =====================

#该文件是项目的入口，负责训练流程的控制，包括数据加载、模型训练、主动学习策略的应用等
def validate_config(cfg):
    """验证配置的合理性"""
    # 1) kappa关系验证
    if cfg["kappa_max"] <= cfg["kappa_min"] * 4:
        # 使用警告而不是断言，避免程序中断
        print(f"警告：kappa范围可能太小: {cfg['kappa_min']}->{cfg['kappa_max']}")
        # 自动修复：确保至少有8倍范围
        cfg["kappa_max"] = cfg["kappa_min"] * 8
        if cfg["kappa_max"] > 256:
            cfg["kappa_max"] = 256

    # 2) 权重非负
    for key in ['w_pan', 'w_dualsoftmax', 'w_lifted', 'w_triplet', 'w_contrastive']:
        if cfg[key] < 0:
            print(f"警告：{key}为负值: {cfg[key]}，自动修正为0")
            cfg[key] = 0.0

    # 3) 温度合理
    if cfg["train_weight_temperature"] < 0.1 or cfg["train_weight_temperature"] > 5.0:
        print(f"警告：温度参数不合理: {cfg['train_weight_temperature']}，自动修正为1.0")
        cfg["train_weight_temperature"] = 1.0

    # 4) 学习率合理范围
    if cfg["learning_rate"] < 1e-5 or cfg["learning_rate"] > 1e-2:
        print(f"警告：学习率超出合理范围: {cfg['learning_rate']}，自动修正为1e-4")
        cfg["learning_rate"] = 1e-4

    return True


def get_dataset_specific_params(dataset_name):
    """
    根据数据集名称动态调整超参数范围。
    """
    if dataset_name == 'MS-COCO':
        return {
            'kappa_min': 2.0, 'kappa_max': 512, 'c_intra': 0.05,
            'lambda_density_range': [0.5, 0.8, 1.2, 1.5, 2.0],
            'lambda_bias_range': [0.5, 0.8, 1.2, 1.5, 2.0],
            'alpha_sample_range': [0.4, 0.6, 0.8, 1.0],
            'train_weight_scale_range': [0.8, 1.2],
            'density_warmup_rounds_range': [5, 7, 10],
            'kappa_soft_tau_range': [0.12, 0.18, 0.25, 0.3],
        }
    elif dataset_name == 'mirflickr':
        return {
            'kappa_min': 1.0, 'kappa_max': 64, 'c_intra': 0.08,
            'lambda_density_range': [0.2, 0.4, 0.6],
            'lambda_bias_range': [0.2, 0.4, 0.6],
            'alpha_sample_range': [0.4, 0.5, 0.6],
            'train_weight_scale_range': [0.5, 0.8],
            'density_warmup_rounds_range': [2, 3],
            'kappa_soft_tau_range': [0.05, 0.08, 0.12],
        }
    elif dataset_name == 'NUS-WIDE-TC21':
        return {
            'kappa_min': 2.0, 'kappa_max': 512, 'c_intra': 0.08,
            'lambda_density_range': [0.4, 0.6, 0.8, 1.2, 1.5],
            'lambda_bias_range': [0.4, 0.6, 0.8, 1.2, 1.5],
            'alpha_sample_range': [0.5, 0.6, 0.7, 0.8],
            'train_weight_scale_range': [0.8, 1.0],
            'density_warmup_rounds_range': [3, 5, 7],
            'kappa_soft_tau_range': [0.08, 0.12, 0.18, 0.25],
        }
    else:
        return {
            'kappa_min': 1.0, 'kappa_max': 128, 'c_intra': 0.1,
            'lambda_density_range': [0.3, 0.5, 0.7],
            'lambda_bias_range': [0.3, 0.5, 0.7],
            'alpha_sample_range': [0.4, 0.5, 0.6],
            'train_weight_scale_range': [0.8, 1.2],
            'density_warmup_rounds_range': [3, 5],
            'kappa_soft_tau_range': [0.1, 0.12, 0.15],
        }


# 找到现有的 generate_hyper_configs 函数（大概在第200行左右）
# 将整个函数替换为：

def generate_hyper_configs(seed=0, n_configs=200, dataset_name="MS-COCO"):  # 注意：n_configs从50增加到200
    """生成更针对性的超参数配置"""
    rng = np.random.RandomState(seed)
    configs = []

    for config_id in range(n_configs):
        print(f"生成配置 {config_id + 1}/{n_configs}")

        # 根据数据集选择基础参数
        if dataset_name == 'MS-COCO':
            # COCO需要更强的正则化和更大的学习率范围
            base_config = {
                "learning_rate": float(10 ** rng.uniform(-3.8, -3.0)),  # [1.6e-4, 1e-3]
                "kappa_min": float(rng.choice([0.5, 1.0, 2.0])),
                "kappa_max": float(rng.choice([256, 512])),
                "c_intra": float(rng.choice([0.03, 0.05, 0.08])),
                "lambda_density": float(rng.choice([0.8, 1.2, 1.5, 2.0])),
                "lambda_bias": float(rng.choice([0.5, 0.8, 1.2])),
                "alpha_sample": float(rng.choice([0.4, 0.5, 0.6])),
                "w_pan": float(rng.choice([1.0, 2.0, 4.0, 8.0])),
                "w_dualsoftmax": float(rng.choice([0.5, 1.0, 2.0])),
                "grad_clip": float(rng.choice([1.0, 2.0, 3.0])),
            }
        elif dataset_name == 'NUS-WIDE-TC21':
            # NUS-WIDE：中等设置
            base_config = {
                "learning_rate": float(10 ** rng.uniform(-4.0, -3.2)),  # [1e-4, 6.3e-4]
                "kappa_min": float(rng.choice([1.0, 2.0, 4.0])),
                "kappa_max": float(rng.choice([128, 256])),
                "c_intra": float(rng.choice([0.05, 0.08, 0.12])),
                "lambda_density": float(rng.choice([0.6, 0.8, 1.0, 1.2])),
                "lambda_bias": float(rng.choice([0.4, 0.6, 0.8])),
                "alpha_sample": float(rng.choice([0.5, 0.6, 0.7])),
                "w_pan": float(rng.choice([0.5, 1.0, 2.0, 4.0])),
                "w_dualsoftmax": float(rng.choice([0.5, 1.0, 1.5])),
                "grad_clip": float(rng.choice([0.5, 1.0, 2.0])),
            }
        else:  # mirflickr
            base_config = {
                "learning_rate": float(10 ** rng.uniform(-4.2, -3.5)),  # [6.3e-5, 3.2e-4]
                "kappa_min": float(rng.choice([2.0, 4.0])),
                "kappa_max": float(rng.choice([64, 128])),
                "c_intra": float(rng.choice([0.08, 0.12, 0.15])),
                "lambda_density": float(rng.choice([0.2, 0.4, 0.6])),
                "lambda_bias": float(rng.choice([0.2, 0.4, 0.6])),
                "alpha_sample": float(rng.choice([0.4, 0.5, 0.6])),
                "w_pan": float(rng.choice([0.5, 1.0, 2.0])),
                "w_dualsoftmax": float(rng.choice([0.5, 1.0, 1.5])),
                "grad_clip": float(rng.choice([0.5, 1.0, 1.5])),
            }

        # 添加通用参数
        config = {
            **base_config,
            "kappa_soft_tau": float(rng.choice([0.05, 0.08, 0.12, 0.18])),
            "density_warmup_rounds": int(rng.choice([0, 3, 5, 7])),
            "train_weight_scale": float(rng.choice([0.5, 0.8, 1.0, 1.2])),
            "train_weight_alpha": float(rng.choice([0.5, 1.0, 1.5])),
            "train_weight_beta": float(rng.choice([0.5, 1.0, 1.5])),
            "train_weight_gamma": float(rng.choice([0.5, 1.0, 1.5])),
            "train_weight_temperature": float(rng.choice([0.8, 1.0, 1.2, 1.5])),
            "w_lifted": 0.0,
            "w_triplet": 0.0,
            "w_contrastive": 0.0,
        }

        # 30%的概率添加辅助损失
        if rng.random() < 0.3:
            aux_type = rng.choice(['lifted', 'triplet', 'contrastive'])
            aux_weight = rng.choice([0.1, 0.25, 0.5])
            config[f"w_{aux_type}"] = aux_weight

        # 验证配置
        validate_config(config)
        configs.append(config)

    return configs

def to_jsonable(obj):
    """把 numpy 标量/数组递归转换为 Python 可 JSON 序列化的类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj



font_size = 25
font = {'family' : 'serif',
        'size'   : font_size}

matplotlib.rc('font', **font)

#所有策略
ALL_STRATEGIES = [
    'RandomSampling',
    'EntropySampling',
    'BALDDropout',
    'CoreSet',
    'CDALSampling',
    'UncertaintySampling',
    'ContrastiveSampling',
    'AdversarialDeepFool',
    'GCNSampling',
]

#存储各数据集在“全监督训练”下的参考 mAP
SUPERVISED_MAP = {'MS-COCO': 0.813,
                  'mirflickr': 0.811,
                  'NUS-WIDE-TC21': 0.757}

# 用于进行传统的全监督训练
def supervised_learning(args):
    # supervised training args
    train_parser = argparse.ArgumentParser(description="Training parser for training hyper-parrameters at ech checkpoint.")

    train_parser.add_argument('--data_augmentation', action='store_const', default=False, const=True)
    train_parser.add_argument('--n_epoch', type=int, default=200)
    train_parser.add_argument('--n_early_stopping', type=int, default=200)
    train_parser.add_argument('--optimizer', type=str, default='Adam')
    train_parser.add_argument('--batch_size', type=int, default=64)
    train_parser.add_argument('--learning_rate', type=float, default=0.01)
    train_parser.add_argument('--momentum', type=float, default=0.9)
    train_parser.add_argument('--weight_decay', type=float, default=0.0005)
    train_parser.add_argument('--lr_warmup', type=int, default=0)
    train_parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=None)
    train_parser.add_argument('--lr_schedule', action="store_const", default=False, const=True)
    train_parser.add_argument('--lr_T_0', type=int, default=200)
    train_parser.add_argument('--lr_T_mult', type=int, default=1)
    train_parser.add_argument('--train_to_end', action="store_const", default=False, const=True)
    train_parser.add_argument('--n_validation_set', type=int, default=2000)
    train_parser.add_argument('--choose_best_val_model', action="store_const", default=False, const=True)

    train_parser.add_argument('--model', type=str, default='resnet50')
    train_parser.add_argument('--emb_size', type=int, default=256)
    train_parser.add_argument('--dropout', type=float, default=0)
    train_parser.add_argument('--fine_tune_layers', type=int, default=1)
    train_parser.add_argument('--pretrained_model', action='store_const', default=False, const=True)
    train_parser.add_argument('--continue_training', action='store_const', default=False, const=True)

    train_parser.add_argument('--vit_patch_size', type=int, default=16)
    train_parser.add_argument('--vit_n_last_blocks', type=int, default=4)
    train_parser.add_argument('--vit_avgpool_patchtokens', action='store_const', default=False, const=True)
    train_parser.add_argument('--vit_pretrained_weights', type=str, default='./pretrained_models/dino_vitbase16_pretrain.pth')
    train_parser.add_argument('--early_stop_times', type=int, default=10)

    # 解析命令行参数并存储在 train_args 中，这些超参数将用于模型训练
    train_args, _ = train_parser.parse_known_args()
    
    #统一管理不同数据集的训练超参数和数据加载设置 定义了不同数据集（如 MNIST）的训练超参数，包括数据集的大小、标签数、图像尺寸、数据加载器、优化器设置等
    train_params_pool = {
        'MNIST':
            {'n_epoch': train_args.n_epoch,
             'n_training_set': 60000,
             'n_label': 10,
             'image_size': 28,
             'in_channels': 1,
             'transform': transforms.Compose(
                 [
                     #transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))
                 ]),
             'test_transform': transforms.Compose(
                 [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/MNIST',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training
             },
        'EMNIST':
            {'n_epoch': train_args.n_epoch,
             'n_training_set': 130000,
             'n_label': 26,
             'image_size': 28,
             'in_channels': 1,
             'transform': transforms.Compose(
                 [
                     #transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
             'test_transform': transforms.Compose(
                 [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/EMNIST',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training
             },
        'SVHN':
            {'n_epoch': train_args.n_epoch,
             'n_label': 10,
             'n_training_set': 60000,
             'image_size': 32,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     #transforms.RandomCrop(32, padding=4),
                     #transforms.RandomHorizontalFlip(),
                     #transforms.RandomRotation(60),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                 ]
             ),
             'test_transform': transforms.Compose(
                 [
                     transforms.ToTensor(),
                     transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                 ]
             ),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/SVHN',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'CIFAR10':
            {'n_epoch': train_args.n_epoch,
             'n_label': 10,
             'n_training_set': 60000,
             'image_size': 32,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 ] if train_args.data_augmentation else
                 [
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 ]
             ),
             'test_transform': transforms.Compose(
                 [
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 ]
             ),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             # 'optimizer_args': {'lr': 0.05, 'momentum': 0.9},
             'optimizer_args': {'lr': train_args.learning_rate},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/CIFAR10',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'CIFAR100':
            {'n_epoch': train_args.n_epoch,
             'n_label': 100,
             'n_training_set': 50000,
             'image_size': 32,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 ] if train_args.data_augmentation else
                 [
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 ]),
             'test_transform': transforms.Compose(
                 [
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 ]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/CIFAR100',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'MiniImageNet':
            {'n_epoch': train_args.n_epoch,
             'n_label': 100,
             'n_training_set': 50000,
             'image_size': 84,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     #transforms.Resize((224, 224)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]
                 if train_args.data_augmentation else
                 [
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'test_transform': transforms.Compose(
                 [
                     #transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},    #, 'weight_decay': 5e-4},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/MiniImageNet',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'domain_net-real':
            {'n_epoch': train_args.n_epoch,
             'n_label': 345,
             'n_training_set': 130000,
             'image_size': 224,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     transforms.Resize((224, 224)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]
                 if train_args.data_augmentation else
                 [
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'test_transform': transforms.Compose(
                 [
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},    #, 'weight_decay': 5e-4},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/domain_net-real',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'mini_domain_net-real':
            {'n_epoch': train_args.n_epoch,
             'n_label': 20,
             'n_training_set': 8286,
             'image_size': 224,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     transforms.Resize((224, 224)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]
                 if train_args.data_augmentation else
                 [
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'test_transform': transforms.Compose(
                 [
                     #transforms.Resize((84, 84)),
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},    #, 'weight_decay': 5e-4},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/mini_domain_net-real',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'tiny_domain_net-real':
            {'n_epoch': train_args.n_epoch,
             'n_label': 10,
             'n_training_set': 5000,
             'image_size': 224,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     transforms.Resize((224, 224)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]
                 if train_args.data_augmentation else
                 [
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'test_transform': transforms.Compose(
                 [
                     #transforms.Resize((84, 84)),
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},    #, 'weight_decay': 5e-4},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/mini_domain_net-real',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'openml':
            {'n_epoch': train_args.n_epoch,
             'n_training_set': 50000,
             'in_channels': 1,
             'loader_tr_args': {'batch_size': 128, 'num_workers': 1},
             'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
             'optimizer_args': {'lr': train_args.learning_rate},
             'transform': transforms.Compose([transforms.ToTensor()]),
             'test_transform': transforms.Compose([transforms.ToTensor()]),
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'mirflickr':
            {'n_epoch': train_args.n_epoch,
             "alpha": 0.3,
             "beta": 0.2,
             "max_epoch": 40,
             "batch_size": 100,
             "betas": (0.5, 0.999),
             "t": 0.4,
             "gnn": 'GCN',
             "n_layers": 5,
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 4,"pin_memory": True},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 2,"pin_memory": True},
             'optimizer_args': {'lr': train_args.learning_rate},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/mirflickr',
             'n_early_stopping': train_args.n_early_stopping,
             'early_stop_times': train_args.early_stop_times,
             'continue_training': train_args.continue_training,
             'task': args.task,
             'grad_clip': 1.0,  # 梯度裁剪阈值
             'lr_schedule': True,  # 启用学习率调度
             'scheduler_type': 'cosine',  # 'cosine'|'step'|'reduce_on_plateau'
             'T_max': 50,  # 余弦退火的周期
             'eta_min': 1e-6,  # 最小学习率
             'early_stop_patience': 10,  # 早停耐心值
             'early_stop_min_delta': 1e-4,  # 最小改进阈值
             },
        'NUS-WIDE-TC21':
            {'n_epoch': train_args.n_epoch,
             "alpha": 0.2,
             "beta": 0.3,
             "max_epoch": 40,
             "batch_size": 100,
             "betas": (0.5, 0.999),
             "t": 0.3,
             "gnn": 'GAT',
             "n_layers": 5,
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 4,"pin_memory": True},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 2,"pin_memory": True},
             'optimizer_args': {'lr': train_args.learning_rate},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/NUS-WIDE-TC21',
             'n_early_stopping': train_args.n_early_stopping,
             'early_stop_times': train_args.early_stop_times,
             'continue_training': train_args.continue_training,
             'task': args.task},
        'MS-COCO':
            {'n_epoch': train_args.n_epoch,
             "alpha": 2.8,
             "beta": 0.2,
             "max_epoch": 40,
             "batch_size": 512,
             "betas": (0.5, 0.999),
             "t": 0.2,
             'k': 8,
             'gamma': 0.14,
             "gnn": 'GAT',
             "n_layers": 5,
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 4,"pin_memory": True},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 2,"pin_memory": True},
             'optimizer_args': {'lr': train_args.learning_rate},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/NUS-WIDE-TC21',
             'n_early_stopping': train_args.n_early_stopping,
             'early_stop_times': train_args.early_stop_times,
             'continue_training': train_args.continue_training,
             'task': args.task}

    }

    if is_openml(args.data_name):
        train_params = train_params_pool['openml']
    else:
        train_params = train_params_pool[args.data_name]

    train_params['optimizer'] = train_args.optimizer
    if train_args.optimizer == 'SGD':
        train_params['optimizer_args']['momentum'] = train_args.momentum
    train_params['lr_warmup'] = train_args.lr_warmup
    train_params['lr_schedule'] = train_args.lr_schedule
    train_params['lr_T_0'] = train_args.lr_T_0
    train_params['lr_T_mult'] = train_args.lr_T_mult
    train_params['choose_best_val_model'] = train_args.choose_best_val_model

    # 如果 args.strategy 设置为 'All'，则对所有策略（ALL_STRATEGIES）进行循环训练，每个策略都调用 al_train()
    if args.strategy == 'All':
        for strategy in ALL_STRATEGIES:
            al_train(args, train_args, train_params, strategy)
    else:
        al_train(args, train_args, train_params, args.strategy)

# 函数用于管理一个完整的主动学习实验，从数据准备、模型初始化到训练过程的每个步骤。它负责为每种策略创建实验目录，并调用子实验函数
# def al_train(args, train_args, train_params, strategy_name):
#     #根据指定的日志目录和数据集名称，生成实验的主目录。若目录不存在，则创建该目录
#     main_path = os.path.join(args.log_dir, args.data_name)
#     if not os.path.exists(main_path):
#         os.makedirs(main_path)
#     # 生成实验路径：这里将多个训练配置（如初始标注样本数、查询样本数、批量大小、学习率等）组合成一个实验目录的路径。每个实验的配置都会生成一个唯一的目录，用于保存实验结果和模型。
#     general_path = os.path.join(main_path,
#                                 'init' + str(args.n_init_lb) + '_query' + str(args.n_query) + '_' + str(args.query_growth_ratio) +
#                                 '_rounds' + str(args.n_round) + '_' + train_args.model + '_emb' + str(train_args.emb_size) +
#                                 '_bs' + str(train_args.batch_size) + ('_augmentation' if train_args.data_augmentation else '') +
#                                 '_epochs' + str(train_args.n_epoch) + ('_full' if train_args.train_to_end else '') +
#                                 ('_continue' if train_args.continue_training else '') +
#                                 ('' if train_args.model[:3] != 'vit' else ('_patch' + str(train_args.vit_patch_size) + '_nblock' + str(train_args.vit_n_last_blocks) + ('_avgpool' if train_args.vit_avgpool_patchtokens else ''))) +
#                                 (('_pretrained' + str(train_args.fine_tune_layers)) if train_args.pretrained_model else '') +
#                                 ('_lr_schedule' if train_args.lr_schedule else '') +
#                                 (('_lr_warmup' + str(train_args.lr_warmup)) if train_args.lr_warmup > 0 else '') +
#                                 '_dropout' + str(train_args.dropout) +
#                                 '_lr' + str(train_args.learning_rate) +
#                                 ('' if train_args.lr_decay_epochs is None else ('_lrd' + '-'.join([str(e) for e in train_args.lr_decay_epochs]))) +
#                                 '_valid' + str(train_args.n_validation_set) +
#                                 '_es' + str(train_args.early_stop_times) +
#                                 '_task' + args.task + '_K' + str(args.K))
#     # 创建实验子目录：如果该路径不存在，创建它。这是用来存放每个具体实验（如每个随机种子、每个策略）的结果。
#     if not os.path.exists(general_path):
#         os.makedirs(general_path)
#
#     # 循环执行子实验：根据不同的随机种子（args.seeds），为每个种子创建一个子实验，并调用 al_train_sub_experiment() 来执行完整的实验流程
#     #for seed in args.seeds:
#     # for init_label_num in [100, 1000, 10000]:
#     #     args.n_init_lb = init_label_num
#     seed = args.seeds[0]
#     #al_train_sub_experiment(args, train_args, train_params, strategy_name, general_path, args.seeds[0])
#     #超参数调优模块---目前先不做
#     # 假设这里的 train_params 是你从 train_params_pool 取出来的“基准配置”
#     base_train_params = deepcopy(train_params)
#     hyper_configs = generate_hyper_configs(seed)
#     for hyper_id, hyper_cfg in enumerate(hyper_configs):
#
#         print(f"\n========== Hyper Config {hyper_id} ==========")
#         print(hyper_cfg)
#
#         # 1) 绑定 hyper 信息（用于日志/落盘）
#         args.hyper_id = hyper_id
#         args.hyper_config = hyper_cfg
#
#         # 2) 为这个 hyper config 创建独立目录（避免覆盖）
#         hyper_path = os.path.join(general_path, f"hyper_{hyper_id}")
#         os.makedirs(hyper_path, exist_ok=True)
#
#         # 3) 保存可追溯超参文件（hyper_id -> hyper_config.json）
#         with open(os.path.join(hyper_path, "hyper_config.json"), "w") as f:
#             json.dump(to_jsonable(hyper_cfg), f, indent=2)
#
#         # 4) 注入“主动学习/采样”相关参数到 args（VMFSampling 用 argget(args, ...) 读）
#         #    这里直接把 hyper_cfg 全部 setattr 进去没问题；训练器不读 args
#         for k, v in hyper_cfg.items():
#             setattr(args, k, v)
#
#         # 5) 注入“训练器”相关参数到 train_params（CLIP4CMRTraining 从 dict 里读）
#         #    注意：用本轮独立 train_params，避免污染下一轮 hyper cfg
#         train_params = deepcopy(base_train_params)
#         train_params.update({
#             "learning_rate": hyper_cfg["learning_rate"],
#             "w_pan": hyper_cfg["w_pan"],
#             "w_dualsoftmax": hyper_cfg["w_dualsoftmax"],
#             "grad_clip": hyper_cfg["grad_clip"],  # ★你 cfg 里有，就必须更新进来
#         })
#
#         # 6) 让训练器能把 epoch/round 指标写到这个目录（你后面记录 CSV 会用到）
#         #    main.py 里 train_params_pool 本来就有 log_dir，但我们这里覆盖成每组独立目录更合理
#         train_params["log_dir"] = hyper_path
#
#         # 7) 执行一次完整实验：一组超参 = 一次 al_train_sub_experiment（你的真实入口）
#         al_train_sub_experiment(
#             args,
#             train_args,
#             train_params,
#             strategy_name,
#             hyper_path,
#             seed
#         )


def al_train(args, train_args, train_params, strategy_name):
    #根据指定的日志目录和数据集名称，生成实验的主目录。若目录不存在，则创建该目录
    main_path = os.path.join(args.log_dir, args.data_name)
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    # 生成实验路径：这里将多个训练配置（如初始标注样本数、查询样本数、批量大小、学习率等）组合成一个实验目录的路径。每个实验的配置都会生成一个唯一的目录，用于保存实验结果和模型。
    general_path = os.path.join(main_path,
                                'init' + str(args.n_init_lb) + '_query' + str(args.n_query) + '_' + str(args.query_growth_ratio) +
                                '_rounds' + str(args.n_round) + '_' + train_args.model + '_emb' + str(train_args.emb_size) +
                                '_bs' + str(train_args.batch_size) + ('_augmentation' if train_args.data_augmentation else '') +
                                '_epochs' + str(train_args.n_epoch) + ('_full' if train_args.train_to_end else '') +
                                ('_continue' if train_args.continue_training else '') +
                                ('' if train_args.model[:3] != 'vit' else ('_patch' + str(train_args.vit_patch_size) + '_nblock' + str(train_args.vit_n_last_blocks) + ('_avgpool' if train_args.vit_avgpool_patchtokens else ''))) +
                                (('_pretrained' + str(train_args.fine_tune_layers)) if train_args.pretrained_model else '') +
                                ('_lr_schedule' if train_args.lr_schedule else '') +
                                (('_lr_warmup' + str(train_args.lr_warmup)) if train_args.lr_warmup > 0 else '') +
                                '_dropout' + str(train_args.dropout) +
                                '_lr' + str(train_args.learning_rate) +
                                ('' if train_args.lr_decay_epochs is None else ('_lrd' + '-'.join([str(e) for e in train_args.lr_decay_epochs]))) +
                                '_valid' + str(train_args.n_validation_set) +
                                '_es' + str(train_args.early_stop_times) +
                                '_task' + args.task + '_K' + str(args.K)+f'_{strategy_name}'+
                                ('_biascalibration' if args.enable_bias_calibration else '') +
                                ('_diversityselection' if args.enable_diversity_selection else '') +
                                ('_disabletrainweight' if args.disable_train_weight else '') +
                                ('_fixedpool' if str(getattr(args, 'fixed_query_dir', '') or '').strip() else ''))
    # 创建实验子目录：如果该路径不存在，创建它。这是用来存放每个具体实验（如每个随机种子、每个策略）的结果。
    if not os.path.exists(general_path):
        os.makedirs(general_path)

    # 循环执行子实验：根据不同的随机种子（args.seeds），为每个种子创建一个子实验，并调用 al_train_sub_experiment() 来执行完整的实验流程
    #for seed in args.seeds:
    # for init_label_num in [100, 1000, 10000]:
    #     args.n_init_lb = init_label_num
    # 多种子支持：seeds_to_run 保存所有要跑的种子
    seeds_to_run = list(args.seeds)   # e.g. [1, 10, 100]
    seed = seeds_to_run[0]            # 兼容 sweep/single_id 模式
    #al_train_sub_experiment(args, train_args, train_params, strategy_name, general_path, args.seeds[0])
    #超参数调优模块---目前先不做
    # 假设这里的 train_params 是你从 train_params_pool 取出来的“基准配置”
    base_train_params = deepcopy(train_params)

    # ==========================================================
    # 选择运行模式：
    # 1) --hyper_config_path 指定 json：只跑这个 json 配置
    # 2) --hyper_id >= 0：只跑 generate_hyper_configs(seed) 中对应 id
    # 3) 默认：跑所有组合（sweep）
    # 并且不同模式输出到不同目录，避免日志覆盖
    # ==========================================================
    run_mode = "sweep"  # 默认全遍历
    selected_hyper_id = None

    # 读取命令行参数（args 可能是 Namespace）
    hyper_config_path = getattr(args, "hyper_config_path", "") or ""
    hyper_id_cli = int(getattr(args, "hyper_id", -1))

    if hyper_config_path.strip() != "":
        # ---- 模式 1：从 json 读单一配置 ----
        run_mode = "single_json"
        with open(hyper_config_path, "r") as f:
            one_cfg = json.load(f)
        hyper_iter = [(0, one_cfg)]  # 这里的 0 是 local id；真正 hyper_id 我们下面会重设
        selected_hyper_id = hyper_id_cli if hyper_id_cli >= 0 else 0

    elif hyper_id_cli >= 0:
        # ---- 模式 2：只跑指定 hyper_id ----
        run_mode = "single_id"
        all_cfgs = generate_hyper_configs(seed,50,dataset)
        if not (0 <= hyper_id_cli < len(all_cfgs)):
            raise ValueError(f"--hyper_id {hyper_id_cli} out of range (0~{len(all_cfgs) - 1})")
        hyper_iter = [(hyper_id_cli, all_cfgs[hyper_id_cli])]
        selected_hyper_id = hyper_id_cli

    else:
        # ---- 模式 3：默认全遍历 ----
        run_mode = "sweep"
        all_cfgs = generate_hyper_configs(seed,50,dataset)
        hyper_iter = list(enumerate(all_cfgs))

    # 不同模式写到不同目录，避免覆盖
    mode_path = os.path.join(general_path, run_mode)
    os.makedirs(mode_path, exist_ok=True)

    # ===================== NEW: two-stage sweep =====================
    if run_mode == "sweep" and bool(getattr(args, "sweep_two_stage", False)):
        all_cfgs = generate_hyper_configs(seed,50,dataset)
        # -------- Stage-1 screening --------
        stage1_path = os.path.join(mode_path, "stage1_screen")
        os.makedirs(stage1_path, exist_ok=True)

        # 备份原预算
        full_n_round = int(args.n_round)
        full_n_epoch = int(base_train_params.get("n_epoch", train_args.n_epoch))

        screen_rounds = int(getattr(args, "sweep_screen_rounds", 4))
        screen_epoch = int(getattr(args, "sweep_screen_epoch", 2))
        topk = int(getattr(args, "sweep_topk", 10))
        metric_col = str(getattr(args, "sweep_score_metric", "test_mAP"))
        score_mode = str(getattr(args, "sweep_score_mode", "stable"))

        scores = []  # (score, hyper_id)

        for hyper_id, hyper_cfg in enumerate(all_cfgs):
            print(f"\n[Stage-1] Hyper {hyper_id}/{len(all_cfgs) - 1}")
            # 目录
            hyper_path = os.path.join(stage1_path, f"hyper_{hyper_id}")
            os.makedirs(hyper_path, exist_ok=True)

            # 注入 args
            args.hyper_id = int(hyper_id)
            args.hyper_config = hyper_cfg
            for k, v in hyper_cfg.items():
                setattr(args, k, v)

            # 用小预算
            args.n_round = int(screen_rounds)

            train_params_this = deepcopy(base_train_params)
            train_params_this.update({
                "n_epoch": int(screen_epoch),
                "learning_rate": float(hyper_cfg["learning_rate"]),
                "w_pan": float(hyper_cfg["w_pan"]),
                "w_dualsoftmax": float(hyper_cfg["w_dualsoftmax"]),
                "w_lifted": float(hyper_cfg.get("w_lifted", 0.0)),
                "w_triplet": float(hyper_cfg.get("w_triplet", 0.0)),
                "w_contrastive": float(hyper_cfg.get("w_contrastive", 0.0)),
                "grad_clip": float(hyper_cfg["grad_clip"]),
                "hyper_id": int(hyper_id),
                "log_dir": hyper_path,
            })

            # 跑一次 screening
            al_train_sub_experiment(args, train_args, train_params_this, strategy_name, hyper_path, seed)

            # 读 round_metrics.csv 评分
            round_csv = os.path.join(hyper_path, f"{strategy_name}_seed{seed}", "round_metrics.csv")
            try:
                st = compute_al_curve_stats(round_csv, metric_col=metric_col, stable_alpha=1.0, stable_beta=1.0)
                if score_mode == "stable":
                    score = float(st["stable_score"])
                elif score_mode == "last2":
                    score = float(st["last2"])
                else:
                    score = float(st["last"])
            except Exception as e:
                print(f"[Stage-1] scoring failed for hyper {hyper_id}: {e}")
                score = -1e18

            scores.append((score, hyper_id))

        # 排序并取 Top-K
        scores.sort(key=lambda x: x[0], reverse=True)
        kept = [hid for _, hid in scores[:max(1, topk)]]
        with open(os.path.join(stage1_path, "ranking_stage1.json"), "w") as f:
            json.dump({"scores": scores, "kept": kept}, f, indent=2)

        print(f"\n[Stage-1] kept top-{len(kept)}: {kept}")

        # -------- Stage-2 full run on Top-K --------
        stage2_path = os.path.join(mode_path, "stage2_full")
        os.makedirs(stage2_path, exist_ok=True)

        # 恢复原预算
        args.n_round = int(full_n_round)

        for hyper_id in kept:
            hyper_cfg = all_cfgs[hyper_id]
            print(f"\n[Stage-2] Hyper {hyper_id} full run")
            hyper_path = os.path.join(stage2_path, f"hyper_{hyper_id}")
            os.makedirs(hyper_path, exist_ok=True)

            args.hyper_id = int(hyper_id)
            args.hyper_config = hyper_cfg
            for k, v in hyper_cfg.items():
                setattr(args, k, v)

            train_params_this = deepcopy(base_train_params)
            train_params_this.update({
                "n_epoch": int(full_n_epoch),
                "learning_rate": float(hyper_cfg["learning_rate"]),
                "w_pan": float(hyper_cfg["w_pan"]),
                "w_dualsoftmax": float(hyper_cfg["w_dualsoftmax"]),
                "w_lifted": float(hyper_cfg.get("w_lifted", 0.0)),
                "w_triplet": float(hyper_cfg.get("w_triplet", 0.0)),
                "w_contrastive": float(hyper_cfg.get("w_contrastive", 0.0)),
                "grad_clip": float(hyper_cfg["grad_clip"]),
                "hyper_id": int(hyper_id),
                "log_dir": hyper_path,
            })

            al_train_sub_experiment(args, train_args, train_params_this, strategy_name, hyper_path, seed)

        # two-stage 做完就 return/exit（避免继续走旧 sweep loop）
        return
    # ===================== end two-stage sweep =====================

    # [FIX] Capture base_fixed_dir ONCE before any hyper/seed loops.
    # Must be outside the for-hyper loop: inner seed loops mutate
    # args.fixed_query_dir, so capturing inside would accumulate paths
    # across hyper iterations (seed_100/seed_10/seed_1 nesting bug).
    base_fixed_dir = str(getattr(args, 'fixed_query_dir', '') or '').strip()

    # ====== Sweep ranking state (do not affect training) ======
    best_stable_score = -1e18
    best_stable_hyper_id = -1
    best_last2_score = -1e18
    best_last2_hyper_id = -1
    # =========================================================

    for hyper_id, hyper_cfg in hyper_iter:

        print(f"\n========== Hyper Config {hyper_id} ({run_mode}) ==========")
        print(hyper_cfg)

        # 1) 绑定 hyper 信息（用于日志/落盘）
        args.hyper_id = int(hyper_id)
        args.hyper_config = hyper_cfg

        # 2) 为这个 hyper config 创建独立目录（避免覆盖）
        hyper_path = os.path.join(mode_path, f"hyper_{hyper_id}")
        os.makedirs(hyper_path, exist_ok=True)

        # 3) 保存可追溯超参文件（hyper_id -> hyper_config.json）
        with open(os.path.join(hyper_path, "hyper_config.json"), "w") as f:
            json.dump(to_jsonable(hyper_cfg), f, indent=2)

        # 4) 注入“主动学习/采样”相关参数到 args（VMFSampling 用 argget(args, ...) 读）
        for k, v in hyper_cfg.items():
            setattr(args, k, v)

        # 5) 注入“训练器”相关参数到 train_params（CLIP4CMRTraining 从 dict 里读）
        train_params_this = deepcopy(base_train_params)
        train_params_this.update({
            "learning_rate": float(hyper_cfg["learning_rate"]),
            "w_pan": float(hyper_cfg["w_pan"]),
            "w_dualsoftmax": float(hyper_cfg["w_dualsoftmax"]),
            "w_lifted": float(hyper_cfg.get("w_lifted", 0.0)),  # 新增
            "w_triplet": float(hyper_cfg.get("w_triplet", 0.0)),  # 新增
            "w_contrastive": float(hyper_cfg.get("w_contrastive", 0.0)),  # 新增
            "grad_clip": float(hyper_cfg["grad_clip"]),
            # 可追溯字段
            "hyper_id": int(hyper_id),
            "log_dir": hyper_path,
        })
        # [关键修复]：记录原始的基础路径，防止在循环中被重复拼接
        # 6) 多种子循环：每个种子各跑一次完整实验
        for _seed in seeds_to_run:
            print(f"\n[MultiSeed] seed={_seed} ({seeds_to_run.index(_seed)+1}/{len(seeds_to_run)})")

            # [FIX] Per-seed isolation for fixed_query_dir.
            # Always set args.fixed_query_dir explicitly (even to '') so
            # previous hyper-iteration's value never bleeds into this run.
            if base_fixed_dir:
                seed_fixed_dir = os.path.join(base_fixed_dir, f'seed_{_seed}')
                os.makedirs(seed_fixed_dir, exist_ok=True)
                args.fixed_query_dir = seed_fixed_dir
                print(f'[PathIsolation] fixed_query_dir={seed_fixed_dir}')
            else:
                args.fixed_query_dir = ''
            al_train_sub_experiment(
                args,
                train_args,
                train_params_this,
                strategy_name,
                hyper_path,
                _seed
            )

        # ====== After this hyper config finishes: summarize learning-curve stats ======
        exp_name = strategy_name + '_seed' + str(seed)
        round_metrics_csv = os.path.join(hyper_path, exp_name, "round_metrics.csv")

        # Choose which column to score on. Default: test_mAP (matches your round_metrics.csv header)
        score_metric = getattr(args, "hyper_score_metric", "test_mAP") or "test_mAP"

        # Stability penalty weights (you can tune later, but 1.0 is a good default)
        stable_alpha = 1.0
        stable_beta = 1.0

        try:
            stats = compute_al_curve_stats(
                round_metrics_csv,
                metric_col=score_metric,
                stable_alpha=stable_alpha,
                stable_beta=stable_beta,
            )
        except Exception as e:
            print(f"[HYPER STATS] hyper_id={hyper_id} failed to compute stats: {e}")
            stats = {
                "mean_all": -1e18,
                "std": 1e18,
                "max_drop": 1e18,
                "stable_score": -1e18,
                "last": -1e18,
                "last2": -1e18,
                "n_rounds": 0,
            }

        # Append to sweep summary (one file per mode_path)
        summary_path = os.path.join(mode_path, "sweep_summary.csv")
        is_new = (not os.path.exists(summary_path))
        with open(summary_path, "a", newline="") as sf:
            sw = csv.writer(sf, quoting=csv.QUOTE_ALL)
            if is_new:
                sw.writerow([
                    "hyper_id", "score_metric",
                    "mean_all", "std", "max_drop", "stable_score",
                    "last", "last2", "n_rounds"
                ])
            sw.writerow([
                int(hyper_id), score_metric,
                float(stats["mean_all"]), float(stats["std"]), float(stats["max_drop"]),
                float(stats["stable_score"]), float(stats["last"]), float(stats["last2"]),
                int(stats["n_rounds"])
            ])

        print(
            f"[HYPER STATS] hyper_id={hyper_id} metric={score_metric} "
            f"mean={stats['mean_all']:.6f} std={stats['std']:.6f} "
            f"max_drop={stats['max_drop']:.6f} stable={stats['stable_score']:.6f} "
            f"last={stats['last']:.6f} last2={stats['last2']:.6f}"
        )

        # Track best "stable+strong" and best "final performance"
        if stats["stable_score"] > best_stable_score:
            best_stable_score = float(stats["stable_score"])
            best_stable_hyper_id = int(hyper_id)

        if stats["last2"] > best_last2_score:
            best_last2_score = float(stats["last2"])
            best_last2_hyper_id = int(hyper_id)

        # Persist best ids (so you can check at any time during sweep)
        with open(os.path.join(mode_path, "best_config.json"), "w") as bf:
            json.dump(to_jsonable({
                "score_metric": score_metric,
                "stable_alpha": stable_alpha,
                "stable_beta": stable_beta,
                "best_stable": {
                    "hyper_id": int(best_stable_hyper_id),
                    "stable_score": float(best_stable_score),
                },
                "best_bestlast2": {
                    "hyper_id": int(best_last2_hyper_id),
                    "last2": float(best_last2_score),
                },
            }), bf, indent=2)
        # ====== End hyper summary ======


#固定随机种子，保证训练可重复
def set_seeds(seed):
    # set seed
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # seed = 103  # 5
    # torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False

# 保存训练参数为 JSON 文件，用于实验记录和复现 将训练配置（如数据集名称、训练超参数等）保存为 JSON 文件，以便后续复现实验。
def save_args(args, path, name):
    config = vars(args)
    with open(os.path.join(path, name + '.json'), 'w') as f:
        hps = {key: val for key, val in config.items() if not isinstance(val, type)}
        json.dump(to_jsonable(hps), f, indent=2)

#al_train_sub_experiment 是真正执行主动学习循环的函数
# 负责 单次实验（一个种子 + 一个主动学习策略）从数据准备、模型初始化到主动学习轮次训练的完整流程。
"""
    args：实验相关参数（如数据集名称、主动学习策略、轮次数、初始标注量等）

    train_args：训练相关参数（模型类型、batch size、学习率、epoch 等）

    train_params：不同数据集训练参数的字典（如 train_params_pool）

    strategy_name：主动学习策略名称（如 uncertainty、random）

    general_path：实验保存主目录

    seed：随机种子，用于保证实验可复现

    核心任务：完成一个种子下的主动学习实验，保存训练过程和结果。
    
    创建实验目录：每个实验（包括每个种子和策略）都会创建一个独立的子目录。

    保存参数：将实验参数（args 和 train_args）保存为 JSON 文件。
    训练过程的记录：将训练过程中的输出打印到文件而不是终端，并通过 SummaryWriter 记录 TensorBoard 数据。
    模型初始化：根据选择的策略初始化模型，确保使用正确的网络结构。
"""
def al_train_sub_experiment(args, train_args, train_params, strategy_name, general_path, seed):
    #为当前实验创建文件夹：包含主动学习策略 + 种子编号
    #用于保存日志、模型和结果文件。
    exp_name = strategy_name + '_seed' + str(seed)

    sub_path = os.path.join(general_path, exp_name)
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    save_args(args, sub_path, 'args')
    save_args(train_args, sub_path, 'train_args')
    #将所有打印信息保存到 logs.txt 文件，而不是终端。
    if args.print_to_file:
        orig_stdout = sys.stdout
        log_file = open(os.path.join(sub_path, 'logs.txt'), 'w')
        sys.stdout = log_file

    writer = SummaryWriter(log_dir=sub_path)
    # ===== Checkpoint: only keep the best model (by val mAP) =====
    best_ckpt_path = os.path.join(sub_path, "model_best.pt")
    best_val_mAP = -float("inf")

    #创建 CSV 文件保存每轮实验结果（mAP、i2t、t2i、查询时间）。
    result_file = open(sub_path + '.csv', 'w')
    result_writer = csv.writer(result_file, quoting=csv.QUOTE_ALL)

    # ①【新增】在函数开头创建
    # round_metrics.csv（带表头）
    # ===================== 新增：round-level 指标落盘 =====================
    round_metrics_path = os.path.join(sub_path, "round_metrics.csv")
    is_new_round_metrics = (not os.path.exists(round_metrics_path))
    round_metrics_f = open(round_metrics_path, "a", newline="")
    round_writer = csv.writer(round_metrics_f, quoting=csv.QUOTE_ALL)

    if is_new_round_metrics:
        round_writer.writerow([
            "hyper_id", "seed", "strategy", "round",
            "labeled_num", "unlabeled_num",
            "budget", "query_time_sec",
            "val_mAP", "val_i2t", "val_t2i",
            "test_mAP", "test_i2t", "test_t2i"
        ])
        round_metrics_f.flush()
    # ===================== 新增结束 =====================

    # set seed 使用dagnn的set_seeds 保证随机初始化、数据打乱和 GPU 计算可复现。
    set_seeds(seed)

    # load dataset
    # ITR 表示跨模态检索任务（Image-Text Retrieval），需要图像和文本特征  X_tr, Y_tr 是训练数据和标签；X_te, Y_te 是测试集
    # if args.task == 'ITR':
    #     X_img_tr, X_txt_tr, Y_tr, X_img_te, X_txt_te, Y_te = get_dataset(args.data_name, args.data_dir)
    #     X_tr = (X_img_tr, X_txt_tr)
    #     X_te = (X_img_te, X_txt_te)
    X_img_val, X_txt_val, Y_val_for_itr = [], [], []
    if args.task == 'ITR':
        X_img_tr, X_txt_tr, Y_tr, X_img_te, X_txt_te, Y_te = get_dataset(args.data_name, args.data_dir)
        X_tr = (X_img_tr, X_txt_tr)
        X_te = (X_img_te, X_txt_te)
        # --- 新增：为 ITR 任务创建验证集 ---
        X_img_val, X_txt_val, Y_val_for_itr = [], [], []
        if train_args.n_validation_set > 0:
            n_total = len(X_img_tr)
            assert train_args.n_validation_set < n_total, "n_validation_set cannot be larger than the training pool!"
            # idxs_tmp = np.arange(n_total)
            # np.random.shuffle(idxs_tmp)
            # val_idxs = idxs_tmp[-train_args.n_validation_set:]
            # trn_idxs = idxs_tmp[:-train_args.n_validation_set]
            # 为了验证集划分合理
            # --- 分桶抽样：按每个样本的正标签数（label density）做粗分层 ---
            y_np_all = np.asarray(Y_tr)  # 确保可用 numpy 计算
            pos_cnt = y_np_all.sum(axis=1).astype(int)

            # 4 个桶：0-1 / 2-3 / 4-6 / 7+
            bins = np.digitize(pos_cnt, bins=[1, 3, 6], right=True)

            idxs_tmp = np.arange(n_total)
            np.random.shuffle(idxs_tmp)

            val_idxs_list = []
            n_val = train_args.n_validation_set
            unique_bins = np.unique(bins)

            for b in unique_bins:
                bucket = idxs_tmp[bins[idxs_tmp] == b]
                if len(bucket) == 0:
                    continue

                k = int(round(n_val * (len(bucket) / n_total)))
                k = min(k, len(bucket))

                # 尽量保证每个非空桶至少抽 1 个（前提是 n_val 足够）
                if k == 0 and n_val >= len(unique_bins):
                    k = 1

                if k > 0:
                    val_idxs_list.append(bucket[:k])

            val_idxs = np.concatenate(val_idxs_list) if val_idxs_list else np.array([], dtype=int)

            # 裁剪前打乱，避免偏向某些桶
            np.random.shuffle(val_idxs)

            # 补齐/裁剪，保证恰好 n_val
            if len(val_idxs) < n_val:
                remain = np.setdiff1d(idxs_tmp, val_idxs, assume_unique=False)
                np.random.shuffle(remain)  # ★关键：消除 setdiff1d 的排序偏差
                val_idxs = np.concatenate([val_idxs, remain[: (n_val - len(val_idxs))]])
            elif len(val_idxs) > n_val:
                val_idxs = val_idxs[:n_val]

            # 训练索引 = 全部 - 验证
            trn_idxs = np.setdiff1d(np.arange(n_total), val_idxs, assume_unique=False)

            X_img_val = X_img_tr[val_idxs]
            X_txt_val = X_txt_tr[val_idxs]
            Y_val_for_itr = Y_tr[val_idxs]
            X_img_tr = X_img_tr[trn_idxs]
            X_txt_tr = X_txt_tr[trn_idxs]
            Y_tr = Y_tr[trn_idxs]
            X_tr = (X_img_tr, X_txt_tr)

            print("VAL size:", len(val_idxs), "TRAIN pool size:", len(trn_idxs))
            print("Val density mean:", y_np_all[val_idxs].sum(axis=1).mean(),
                  "Train density mean:", y_np_all[trn_idxs].sum(axis=1).mean())

    else:
        X_tr, Y_tr, X_te, Y_te = get_dataset(args.data_name, args.data_dir)
    # X_tr: 60000, 28, 28; Y_tr: 60000  根据数据集调整训练参数：类别数、embedding 维度、输入维度等
    if is_openml(args.data_name):
        train_params['n_label'] = int(max(Y_tr) + 1)
        train_params['emb_size'] = train_args.emb_size  # 1024
        train_params['dim'] = X_tr.shape[1]
    else:
        train_params['emb_size'] = train_args.emb_size  #256
        if args.task == 'ITR':
            train_params['dim'] = np.shape(X_img_tr)[1:]  # 4096
            train_params['n_label'] = Y_tr.shape[1]  # 4096
        else:
            train_params['dim'] = np.shape(X_tr)[1:]    # (28, 28)

    args.n_label = train_params['n_label']  # 10

    # Generate the validation set 从训练集随机抽取 n_validation_set 个样本作为验证集
    if train_args.n_validation_set > 0 and args.task != 'ITR':
        # shuffle the training data
        idxs_tmp = np.arange(len(X_tr))
        np.random.shuffle(idxs_tmp)
        X_tr = X_tr[idxs_tmp]
        Y_tr = Y_tr[idxs_tmp]

        X_val = X_tr[-train_args.n_validation_set:]
        Y_val = Y_tr[-train_args.n_validation_set:]
    else:
        X_val = []
        Y_val = []

    # 果任务不是跨模态检索（ITR），则根据 n_training_set 限制训练集的大小，避免训练集超过最大大小。
    if args.task != 'ITR':
        X_tr = X_tr[:min(train_params['n_training_set'], len(X_tr) - len(X_val))]
        Y_tr = Y_tr[:min(train_params['n_training_set'], len(Y_tr) - len(Y_val))]

    # start experiment 初始标注池 idxs_lb 标记哪些样本被标注 idxs_tmp 是打乱的训练样本索引
    # n_pool 和 n_test：定义了训练集和测试集的大小，n_pool 是训练样本的总数，n_test 是测试样本的总数。Y_tr 和 Y_te 分别代表训练集和测试集的标签
    n_pool = len(Y_tr)  # 60000 # 18013
    n_test = len(Y_te)  # 10000 # 2002

    # generate initial labeled pool 开始实验时，初始化 标注池（idxs_lb），其中标记哪些样本已经被标注 idxs_lb 是一个布尔数组，
    # 用于表示哪些样本已经被标注。首先将所有值设为 False，然后通过随机打乱索引 idxs_tmp 来选择初始标注样本
    idxs_lb = np.zeros(n_pool, dtype=bool)  # label_pool
    idxs_tmp = np.arange(n_pool)
    np.random.shuffle(idxs_tmp)

    # 这里根据 init_lb_method 来选择初始标注池的生成方法
    # 从训练集随机选择 n_init_lb 个样本进行标注
    if args.init_lb_method == 'general_random':
        if args.n_init_lb < len(idxs_tmp):
            idxs_lb[idxs_tmp[:args.n_init_lb]] = True
        else:
            idxs_lb[idxs_tmp] = True
    elif args.init_lb_method == 'load':
        idxs_lb = np.load(args.lb_init_path)
    else:
        # 其他方法：按类别平衡选择标注样本，确保每个类别都有一定数量的样本
        for i in range(Y_tr.max().item() + 1):
            idx = (Y_tr == i).nonzero().squeeze()
            idxs_lb[idx[:args.n_init_lb]] = True
    # 输出当前标注池、未标注池、验证集和测试集的大小，以及查询预算（即每轮主动学习可选择的样本数量）
    print('Using {} to generate the initialization labelled index'.format(args.init_lb_method))
    print('number of labeled pool: {}'.format(idxs_lb.sum()))
    print('number of unlabeled pool: {}'.format(n_pool - idxs_lb.sum()))
    print('number of validation pool: {}'.format(len(Y_val)))
    print('number of testing pool: {}'.format(n_test))
    print('query budget: {}'.format(args.n_query))
    # 将初始标注池 idxs_lb 保存为 .np 文件，供后续使用
    np.save(open(os.path.join(sub_path, 'query_0.np'), 'wb'), idxs_tmp[idxs_lb])

    # 调用 get_handler() 函数来获取适用于当前数据集的处理器。
    handler = get_handler(args.data_name)
    # 检查当前环境是否支持 GPU，并根据情况选择使用 GPU 或 CPU 进行训练。
    use_cuda = torch.cuda.is_available()
    print("GPU is_available:" + str(torch.cuda.is_available()))
    print('Using %s device.' % ("cuda" if use_cuda else "cpu"))
    device = torch.device("cuda" if use_cuda else "cpu")

    # load network 根据选择的模型类型（如 baseline_cnn），调用相应的网络结构。如果是 baseline_cnn，则调用 get_net() 函数加载网络
    if train_args.model == 'baseline_cnn':
        net = get_net(args.data_name, is_openml(args.data_name))
        net_args = {'n_label': train_params['n_label']}
    elif train_args.model == 'lenet':
        net = LeNet5
        net_args = {'n_label': train_params['n_label']}
    elif train_args.model == 'mlp': # True
        net = MLPNet
        net_args = {'dim': train_params['dim'],
                    'emb_size': train_params['emb_size'],
                    'n_label': train_params['n_label']}
    elif train_args.model == 'dagnn':
        # 加载图神经网络（GNN）：如果选择了 dagnn 模型，则加载图神经网络（GNN）模型，适用于跨模态数据的学习。
        net = DALGNN(img_input_dim=X_img_tr.shape[1], text_input_dim=X_txt_tr.shape[1],
                     num_classes=Y_tr.shape[1], t=t, adj_file='data/' + dataset + '/adj.mat',
                     inp=inp, GNN=gnn, n_layers=n_layers)
    elif train_args.model[:3] == 'vgg':
        net = VGGClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'pretrained': train_args.pretrained_model,
                    'fine_tune_layers': train_args.fine_tune_layers,
                    'emb_size': train_params['emb_size'],
                    'in_channels': train_params['in_channels']}
    elif len(train_args.model) >= 8 and train_args.model[:8] == 'densenet':
        net = DenseNetClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'pretrained': train_args.pretrained_model,
                    'fine_tune_layers': train_args.fine_tune_layers,
                    'emb_size': train_params['emb_size'],
                    'in_channels': train_params['in_channels']}
    elif len(train_args.model) >= 8 and train_args.model[:6] == 'resnet':
        net = ResNetClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'pretrained': train_args.pretrained_model,
                    'fine_tune_layers': train_args.fine_tune_layers,
                    'emb_size': train_params['emb_size'],
                    'in_channels': train_params['in_channels'],
                    'dropout': train_args.dropout}
    # elif train_args.model == 'clip4cmr':
    #     # 添加默认dropout值
    #     dropout_prob = getattr(args, 'dropout', 0.1)  # 默认0.1
    #     net = CLIP4CMRModel(num_class=Y_tr.shape[1],
    #                         img_dim=X_img_tr.shape[1],
    #                         text_dim=X_txt_tr.shape[1],
    #                         mid_dim=256,
    #                         feature_dim=1024,
    #                         dropout_prob=dropout_prob,
    #                         init_weight=True)
    #     model = CLIP4CMRTraining(net, net_args=None, handler=handler, train_params=train_params, writer=writer,
    #                              device=device, init_model=True)
    # --- 跨模态检索骨干网络选择 ---
    if train_args.model in ('clip4cmr', 'albef4cmr', 'vlmo4cmr'):
        dropout_prob = getattr(args, 'dropout', 0.1)

        if train_args.model == 'clip4cmr':
            from itr_models.clip4cmr import CLIP4CMRModel
            net = CLIP4CMRModel(
                num_class=Y_tr.shape[1],
                img_dim=X_img_tr.shape[1],
                text_dim=X_txt_tr.shape[1],
                mid_dim=256,
                feature_dim=1024,
                dropout_prob=dropout_prob,
                init_weight=True
            )
        elif train_args.model == 'albef4cmr':
            from itr_models.albef4cmr import ALBEF4CMRModel
            net = ALBEF4CMRModel(
                num_class=Y_tr.shape[1],
                img_dim=X_img_tr.shape[1],
                text_dim=X_txt_tr.shape[1],
                mid_dim=512,
                feature_dim=1024,
                dropout_prob=dropout_prob,
                init_weight=True,
                n_cross_layers=2,
                n_heads=4
            )
        elif train_args.model == 'vlmo4cmr':
            from itr_models.vlmo4cmr import VLMo4CMRModel
            net = VLMo4CMRModel(
                num_class=Y_tr.shape[1],
                img_dim=X_img_tr.shape[1],
                text_dim=X_txt_tr.shape[1],
                mid_dim=512,
                feature_dim=1024,
                dropout_prob=dropout_prob,
                init_weight=True,
                n_layers=2,
                n_heads=4,
                n_context=4
            )

        model = CLIP4CMRTraining(
            net, net_args=None, handler=handler,
            train_params=train_params, writer=writer,
            device=device, init_model=True
        )

    else:
        # 如果选择了 Vision Transformer（ViT）模型，则加载 VisionTransformerClassifier，并设置相关参数
        net = VisionTransformerClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'pretrained': train_args.pretrained_model,
                    'fine_tune_layers': train_args.fine_tune_layers,
                    'emb_size': train_params['emb_size'],
                    'dropout': train_args.dropout,
                    'patch_size': train_args.vit_patch_size,
                    'n_last_blocks': train_args.vit_n_last_blocks,
                    'avgpool_patchtokens': train_args.vit_avgpool_patchtokens,
                    'pretrained_weights': train_args.vit_pretrained_weights}

    # 根据选择的策略，实例化 CDALModel 或 Training 类。如果选择了 CDALSampling 策略，则使用 CDALModel，否则使用普通的 Training 类
    # if args.strategy == 'CDALSampling':
    #     model = CDALModel(net, None, handler, train_params, writer, device, init_model=True)
    # elif args.strategy == 'VMFSampling':
    #     strategy = VMFSampling(X_tr, Y_tr, idxs_lb, X_val, Y_te, model, args, device, writer, X_img=X_img_tr,
    #                            X_txt=X_txt_tr, X_img_val=X_img_te, X_txt_val=X_txt_te)
    # else:
    #     model = Training(net, None, handler, train_params, writer, device, init_model=True)
    # 统一：先决定 model（训练器）
    # if train_args.model == 'clip4cmr':
    #     # model 已在 clip4cmr 分支里构造好了
    #     pass
    # elif args.strategy == 'CDALSampling':
    #     model = CDALModel(net, None, handler, train_params, writer, device, init_model=True)
    # else:
    #     model = Training(net, None, handler, train_params, writer, device, init_model=True)
    # 统一：先决定 model（训练器）
    if train_args.model in ('clip4cmr', 'albef4cmr', 'vlmo4cmr'):
        # model 已在上面的骨干网络选择分支构造好了
        pass
    elif args.strategy == 'CDALSampling':
        model = CDALModel(net, None, handler, train_params, writer, device, init_model=True)
    else:
        model = Training(net, None, handler, train_params, writer, device, init_model=True)

    # 根据指定的策略名称（如 CDALSampling），选择对应的策略类并实例化
    # 创建策略对象 strategy，传入训练数据（X_tr, Y_tr）、标注池（idxs_lb）、验证集（X_val, Y_te）、
    # 训练的模型 model 等信息。这里创建的是具体的主动学习策略类的实例（例如 CDALSampling 类的实例）
    # cls = globals()[strategy_name]
    # strategy = cls(X_tr, Y_tr, idxs_lb, X_val, Y_te, model, args, device, writer,   # Note Y_te
    #                X_img=X_img_tr, X_txt=X_txt_tr, X_img_val=X_img_te, X_txt_val=X_txt_te)
    cls = globals()[strategy_name]
    # 为 ITR 任务准备正确的验证集参数
    if args.task == 'ITR':
        strategy_X_img_val = X_img_val
        strategy_X_txt_val = X_txt_val
        strategy_Y_val = Y_val_for_itr
    else:
        # 非 ITR 任务保持原样
        strategy_X_img_val = X_img_te
        strategy_X_txt_val = X_txt_te
        strategy_Y_val = Y_val

    # ========== 关键修复：使用纯关键字参数调用 ==========
    strategy = cls(
        X=X_tr,
        Y=Y_tr,
        idxs_lb=idxs_lb,
        X_val=X_val,
        Y_val=strategy_Y_val,  # <-- 统一使用 strategy_Y_val
        model=model,
        args=args,
        device=device,
        writer=writer,
        X_img=X_img_tr,
        X_txt=X_txt_tr,
        X_img_val=strategy_X_img_val,
        X_txt_val=strategy_X_txt_val
    )

    def _inject_round_hyper_to_model_args(strategy, args, rd):
        """
        把 hyper_id 和 round 写入 CLIP4CMRTraining.args (dict)
        这样 epoch_metrics.csv 里能正确记录
        """
        # hyper_id 可能挂在 args (Namespace) 上
        if isinstance(args, dict):
            hyper_id = args.get("hyper_id", -1)
        else:
            hyper_id = getattr(args, "hyper_id", -1)

        if hasattr(strategy, "model") and hasattr(strategy.model, "args"):
            if isinstance(strategy.model.args, dict):
                strategy.model.args["hyper_id"] = int(hyper_id)
                strategy.model.args["round"] = int(rd)
    # ========== 关键修复结束 ==========

    # print info
    print(args.data_name)
    print('SEED {}'.format(seed))
    print(type(strategy).__name__)

    # 用初始标注样本训练模型 使用初始的标注样本（strategy.train(name='0')）进行训练，
    # train() 方法会调用模型进行训练并返回最佳的 mAP、i2t、t2i（mean Average Precision，Image-to-Text 和 Text-to-Image 的检索性能指标）。
    # 用初始标注样本训练模型（Round0）
    strategy.set_round(0)
    _inject_round_hyper_to_model_args(strategy, args, 0)
    best_mAP, best_i2t, best_t2i = strategy.train('0')

    mAP = np.zeros(args.n_round + 1)
    i2t = np.zeros(args.n_round + 1)
    t2i = np.zeros(args.n_round + 1)
    mAP[0] = float(best_mAP[0]) if hasattr(best_mAP, "__len__") else float(best_mAP)
    i2t[0] = float(best_i2t[0]) if hasattr(best_i2t, "__len__") else float(best_i2t)
    t2i[0] = float(best_t2i[0]) if hasattr(best_t2i, "__len__") else float(best_t2i)

    all_test_mAP = np.zeros(args.n_round + 1)
    all_test_i2t = np.zeros(args.n_round + 1)
    all_test_t2i = np.zeros(args.n_round + 1)

    # === 新增：Round0 也评估 testset（与 rd>=1 的逻辑对齐）===
    try:
        current_test_mAP, current_test_i2t, current_test_t2i = strategy.evaluate_on_testset(
            X_img_te=X_img_te, X_txt_te=X_txt_te, Y_te=Y_te
        )
        # 统一保存到 all_test_*[0]，方便你后面画曲线或统计
        all_test_mAP[0] = float(current_test_mAP[0]) if hasattr(current_test_mAP, "__len__") else float(
            current_test_mAP)
        all_test_i2t[0] = float(current_test_i2t[0]) if hasattr(current_test_i2t, "__len__") else float(
            current_test_i2t)
        all_test_t2i[0] = float(current_test_t2i[0]) if hasattr(current_test_t2i, "__len__") else float(
            current_test_t2i)
    except Exception:
        # 若某些设置下不支持/没有 testset，保持兼容（但仍补齐列数）
        all_test_mAP[0], all_test_i2t[0], all_test_t2i[0] = -1.0, -1.0, -1.0

    print('Round 0\ntesting mAP {}'.format(mAP[0]))
    writer.add_scalar('test_accuracy', mAP[0], 0)
    torch.save(strategy.model.clf.state_dict(), os.path.join(sub_path, 'model_round_%d.pt' % (0)))

    # === 修改：result_file 这一行补齐为 7 列（与 rd>=1 对齐）===
    # 原来是: result_writer.writerow([mAP[0], i2t[0], t2i[0], 0.])
    # 现在改成: [val_mAP,val_i2t,val_t2i,test_mAP,test_i2t,test_t2i,duration]
    result_writer.writerow([
        float(mAP[0]), float(i2t[0]), float(t2i[0]),
        float(all_test_mAP[0]), float(all_test_i2t[0]), float(all_test_t2i[0]),
        0.0
    ])
    result_file.flush()

    # ===================== 修改：Round 0 的 round-level 记录（写入真实 test 指标） =====================
    hyper_id = int(args.get("hyper_id", -1)) if isinstance(args, dict) else int(getattr(args, "hyper_id", -1))
    round_writer.writerow([
        hyper_id, seed, type(strategy).__name__, 0,
        int(idxs_lb.sum()), int(n_pool - idxs_lb.sum()),
        0, 0.0,
        float(mAP[0]), float(i2t[0]), float(t2i[0]),
        float(all_test_mAP[0]), float(all_test_i2t[0]), float(all_test_t2i[0])
    ])
    round_metrics_f.flush()
    # ===================== Round0 记录结束 =====================

    # 开始迭代训练，进入循环，执行从第 1 轮到第 args.n_round 轮的主动学习训练，每轮都会进行数据筛选、模型训练、评估等步骤
    for rd in range(1, args.n_round + 1):
        print('Round {}'.format(rd))
        strategy.set_round(rd)

        start_time = time.time()
        #计算查询预算：每轮可标注的样本数量 查询策略：根据不确定性/多样性/混合策略选择要标注的样本
        budget = args.n_query * int(math.pow(args.query_growth_ratio, rd - 1))
        print('query budget: %d' % budget)
        # 调用策略的 query() 方法，选择 budget 数量的样本。
        # q_idxs：选择的样本索引。
        # embeddings：样本的嵌入向量。
        # preds：模型预测的标签。
        # probs：模型对样本的预测概率。
        # u_idxs：未标注样本的索引。
        # candidate_idxs：候选样本索引（当选择策略为混合策略时可能存在）。
        q_idxs, embeddings, preds, probs, u_idxs, candidate_idxs = strategy.query(budget)

        duration = time.time() - start_time
        # query_result 是一个布尔数组，用来标记哪些样本被查询（标注），将选择的样本（q_idxs）标记为 True
        query_result = torch.zeros(Y_tr.shape[0], dtype=torch.bool)
        query_result[q_idxs] = True

        # 查询样本的多样性与不确定性，如果预测结果 preds 不为空
        # s_gt_y：获取查询样本的真实标签。
        # s_y：获取预测标签。
        # s_embeddings：获取查询样本的嵌入向量。
        # s_probs：获取查询样本的预测概率。
        if preds is not None:
            s_gt_y = strategy.Y[q_idxs]

            # [关键修复] 只对查询到的样本 (q_idxs) 计算统计量
            # 而不是所有未标注样本 (u_idxs)
            # q_idxs 通常只有 500 个，u_idxs 有 119500 个
            # 这样 Gram 矩阵 = 500×500 = 2MB，而不是 119500×119500 = 114GB
            s_y = preds[q_idxs]
            s_embddings = embeddings[q_idxs]
            s_probs = probs[q_idxs]

            if isinstance(strategy,
                          (MultiModalSampling, AlphaMixSampling, AdversarialDeepFool, BadgeSampling, CDALSampling,
                           ContrastiveSampling)):
                get_query_diversity_uncertainty(s_embddings, s_gt_y, s_y, s_probs, writer, rd)
        # if preds is not None:
        #     s_gt_y = strategy.Y[q_idxs]
        #     s_y = preds[u_idxs]
        #     s_embddings = embeddings[u_idxs]
        #     s_probs = probs[u_idxs]
        #
        #     if isinstance(strategy,
        #                   (MultiModalSampling, AlphaMixSampling, AdversarialDeepFool, BadgeSampling, CDALSampling,
        #                    ContrastiveSampling)):
        #         get_query_diversity_uncertainty(s_embddings, s_gt_y, s_y, s_probs, writer, rd)

        if args.save_images:
            all_embeds = np.zeros((Y_tr.size()[0], embeddings.size(1)), dtype=float)
            all_embeds[~idxs_lb] = embeddings

            visualise_results(all_embeds, q_idxs if candidate_idxs is None else candidate_idxs, Y_tr, q_idxs,
                              os.path.join(sub_path, 'embedding_round_%d.ckp' % (rd)))
        # 更新标注池 更新标注池：
        # 将当前查询的样本 q_idxs 标记为已标注（True）。
        # 然后调用 strategy.update(idxs_lb) 更新标注池。
        # idxs_lb[q_idxs] = True
        # strategy.update(idxs_lb)
        new_idxs_lb = idxs_lb.copy()
        new_idxs_lb[q_idxs] = True
        strategy.update(new_idxs_lb)
        idxs_lb = new_idxs_lb  # 更新本地引用
        # 用新标注数据重新训练模型 保存模型和本轮查询索引
        print('training with %d labeled samples.' % idxs_lb.sum())
        _inject_round_hyper_to_model_args(strategy, args, rd)
        best_mAP, best_i2t, best_t2i = strategy.train(str(rd))

        mAP[rd] = best_mAP
        i2t[rd] = best_i2t
        t2i[rd] = best_t2i

        # 2. 【关键】在独立的测试集上评估当前模型的真实性能！
        current_test_mAP, current_test_i2t, current_test_t2i = strategy.evaluate_on_testset(X_img_te=X_img_te, X_txt_te=X_txt_te, Y_te=Y_te)
        # 3. 【关键修复】将单轮结果存入预先分配好的数组
        all_test_mAP[rd] = current_test_mAP
        all_test_i2t[rd] = current_test_i2t
        all_test_t2i[rd] = current_test_t2i

        # torch.save(strategy.model.clf.state_dict(), os.path.join(sub_path, 'model_round_%d.pt' % (rd)))
        # Save best checkpoint (val mAP)
        if float(mAP[rd]) > best_val_mAP:
            best_val_mAP = float(mAP[rd])
            torch.save(strategy.model.clf.state_dict(), best_ckpt_path)
            print(f"[Checkpoint] New best model saved: {best_ckpt_path} (val mAP={best_val_mAP:.6f}, round={rd})")
        np.save(open(os.path.join(sub_path, 'query_' + str(rd) + '.np'), 'wb'), q_idxs)

        print('Validateset mAP {}'.format(mAP[rd]))
        writer.add_scalar('Validateset mAP', mAP[rd], rd)
        writer.add_scalar('Validateset i2t_mAP', i2t[rd], rd)
        writer.add_scalar('Validateset t2i_mAP', t2i[rd], rd)

        # 4. 【关键修复】打印和记录时使用正确的数组
        print('Testing mAP {}'.format(all_test_mAP[rd]))
        print('Testing i2t_mAP {}'.format(all_test_i2t[rd]))
        print('Testing t2i_mAP {}'.format(all_test_t2i[rd]))  # 修复了这里的格式化字符串

        writer.add_scalar('Testing mAP', all_test_mAP[rd], rd)
        writer.add_scalar('Testing i2t_mAP', all_test_i2t[rd], rd)
        writer.add_scalar('Testing t2i_mAP', all_test_t2i[rd], rd)
        result_writer.writerow([
            mAP[rd], i2t[rd], t2i[rd],
            all_test_mAP[rd], all_test_i2t[rd], all_test_t2i[rd],
            duration
        ])
        result_file.flush()

        # ===================== 新增：Round rd 的 round-level 记录 =====================
        hyper_id = int(args.get("hyper_id", -1)) if isinstance(args, dict) else int(getattr(args, "hyper_id", -1))
        round_writer.writerow([
            hyper_id, seed, type(strategy).__name__, rd,
            int(idxs_lb.sum()), int(n_pool - idxs_lb.sum()),
            int(budget), float(duration),
            float(mAP[rd]), float(i2t[rd]), float(t2i[rd]),
            float(all_test_mAP[rd]), float(all_test_i2t[rd]), float(all_test_t2i[rd])
        ])
        round_metrics_f.flush()
        # ===================== 新增结束 =====================

        # 如果当前轮 mAP 达到 全监督模型的某个阈值（map_threshold），提前停止实验
        # if best_mAP >= SUPERVISED_MAP[args.data_name] * args.map_threshold:
        #     print('The model exceed {}% mAP of the supervised model'.format(args.map_threshold * 100))
        #     print('The mAP is {}'.format(best_mAP))
        #     print('The round is {}'.format(rd))
        #     print('The label pool size is {}'.format(sum(idxs_lb)))
        #     print('The ratio of labeled data is {}'.format((sum(idxs_lb) + 0.) / (len(idxs_lb))))
        #     writer.add_scalar('ratio_of_labeled_data', (sum(idxs_lb) + 0.) / (len(idxs_lb)), rd)
        #     writer.add_scalar('label_pool_size', sum(idxs_lb), rd)
        #     writer.add_scalar('round', rd, rd)
        #     break

    # print results
    print('SEED {}'.format(seed))
    print(type(strategy).__name__)
    print(mAP)

    # ===== [FIX] 静态消融支持：保存最终标注池布尔索引（种子隔离版）=====
    # Bug fix: original code wrote all seeds to the same path, causing:
    #   (a) seed N overwrites seed N-1's data
    #   (b) shell wait_for_lb_file triggered after seed=1, starting
    #       T1_fixed while T2 was still running seeds 10 and 100.
    # Fix: embed seed into filename so each seed gets its own file.
    _save_lb = str(getattr(args, 'save_final_lb_path', '') or '').strip()
    if _save_lb:
        import os as _os
        _base, _ext = _os.path.splitext(_save_lb)
        _ext = _ext if _ext else '.npy'
        _seed_lb_path = f'{_base}_seed{seed}{_ext}'
        _os.makedirs(_os.path.dirname(_os.path.abspath(_seed_lb_path)), exist_ok=True)
        np.save(_seed_lb_path, idxs_lb)
        print(f'[StaticAblation] idxs_lb saved -> {_seed_lb_path}')
        print(f'[StaticAblation] Labeled count: {int(idxs_lb.sum())} / {len(idxs_lb)}')
    # ===== 保存结束 =====

    # [FIX] Flush TensorBoard writer before close to prevent
    # 'Exception ignored in threading.excepthook' from the async
    # write thread being torn down while still processing events.
    try:
        writer.flush()
        writer.close()
    except Exception as _tb_exc:
        print(f'[TensorBoard] writer close warning (non-fatal): {_tb_exc}')
    result_file.close()

    round_metrics_f.close()

    if args.print_to_file:
        sys.stdout = orig_stdout
        log_file.close()


# def get_query_diversity_uncertainty(embeddings, gt_y, p_y, probs, writer, rd):
#
#     a = np.matmul(embeddings, embeddings.transpose(1, 0))
#     sign, logdet = np.linalg.slogdet(a)
#     print('Log Determinant of the Gram Matrix: %f' % logdet.item())
#     writer.add_scalar('selection_statistics/log_det_gram', logdet.item(), rd)
#
#     print('Signed Log Determinant of the Gram Matrix: %f' % (sign.item() * logdet.item()))
#     writer.add_scalar('selection_statistics/singned_log_det_gram', (sign.item() * logdet.item()), rd)
#
#     conf = probs.max(1)[0].mean().item()
#     print('Confidence: %f' % conf)
#     writer.add_scalar('selection_statistics/confidence', conf, rd)
#
#     probs_sorted, idxs = probs.sort(descending=True)
#     margin = (probs_sorted[:, 0] - probs_sorted[:, 1]).mean().item()
#     print('Margin: %f' % margin)
#     writer.add_scalar('selection_statistics/margin', margin, rd)
#
#     from scipy.stats import entropy
#     p = np.zeros(probs.size(1), dtype=float)
#     for i in range(probs.size(1)):
#         p[i] = (p_y == i).sum().item() / p_y.size(0)
#     ent = entropy(p)
#     print('Predicted Entropy: %f' % ent)
#     writer.add_scalar('selection_statistics/predicted_entropy', ent, rd)
#
#     p = np.zeros(probs.size(1), dtype=float)
#     for i in range(probs.size(1)):
#         p[i] = (gt_y == i).sum().item() / p_y.size(0)
#     ent = entropy(p)
#     print('GT Entropy: %f' % ent)
#     writer.add_scalar('selection_statistics/gt_entropy', ent, rd)
#
#     c = idxs[:, 0:2].min(dim=1)[0] * 1000 + idxs[:, 0:2].max(dim=1)[0]
#     n = int(probs.size(1) * (probs.size(1) - 1) / 2)
#     p = np.zeros(n, dtype=float)
#     idx = 0
#     for i in range(probs.size(1) - 1):
#         for j in range(i + 1, probs.size(1)):
#             p[idx] = (c == (i * 1000) + j).sum().item() / p_y.size(0)
#             idx += 1
#     ent = entropy(p)
#     print('Border Entropy: %f' % ent)
#     writer.add_scalar('selection_statistics/border_entropy', ent, rd)
def ensure_tensor(input_data):
    """确保输入数据是 PyTorch Tensor 类型"""
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)  # 强制转换为 Tensor 类型
    return input_data


def get_query_diversity_uncertainty(embeddings, gt_y, p_y, probs, writer, rd,
                                    max_gram_samples=5000):
    """
    计算查询样本的多样性与不确定性统计量。

    [关键修复] 对大数据集做安全降采样，避免 N×N Gram 矩阵导致内存爆炸。
    COCO 119500 个样本的 Gram 矩阵 = 114GB，直接 OOM。
    降采样到 max_gram_samples=5000 → Gram 仅 200MB，统计量仍有代表性。
    """
    # 确保 p_y 和 probs 是 Tensor 类型
    p_y = ensure_tensor(p_y)
    probs = ensure_tensor(probs)

    N = embeddings.shape[0]

    # --- Gram 矩阵相关统计：大数据集降采样 ---
    if N > max_gram_samples:
        print(f'  [DivStats] N={N} > {max_gram_samples}, subsampling for Gram matrix...')
        rng = np.random.default_rng(rd)
        sub_idx = rng.choice(N, size=max_gram_samples, replace=False)
        emb_sub = embeddings[sub_idx]
    else:
        emb_sub = embeddings

    a = np.matmul(emb_sub, emb_sub.transpose(1, 0))
    # 添加小的正则化项，防止奇异矩阵
    a += np.eye(a.shape[0]) * 1e-6
    sign, logdet = np.linalg.slogdet(a)

    print('Log Determinant of the Gram Matrix: %f' % logdet.item())
    writer.add_scalar('selection_statistics/log_det_gram', logdet.item(), rd)
    print('Signed Log Determinant of the Gram Matrix: %f' % (sign.item() * logdet.item()))
    writer.add_scalar('selection_statistics/singned_log_det_gram', (sign.item() * logdet.item()), rd)

    del a, emb_sub  # 立即释放

    # --- 以下统计不涉及 N×N 矩阵，可以对全量数据计算 ---
    conf = probs.max(1)[0].mean().item()
    print('Confidence: %f' % conf)
    writer.add_scalar('selection_statistics/confidence', conf, rd)

    probs_sorted, idxs = probs.sort(descending=True)
    margin = (probs_sorted[:, 0] - probs_sorted[:, 1]).mean().item()
    print('Margin: %f' % margin)
    writer.add_scalar('selection_statistics/margin', margin, rd)

    from scipy.stats import entropy

    n_classes = probs.size(1)

    # Predicted Entropy
    p = np.zeros(n_classes, dtype=float)
    for i in range(n_classes):
        p[i] = (p_y == i).sum().item() / p_y.size(0)
    ent = entropy(p)
    print('Predicted Entropy: %f' % ent)
    writer.add_scalar('selection_statistics/predicted_entropy', ent, rd)

    # GT Entropy
    p = np.zeros(n_classes, dtype=float)
    for i in range(n_classes):
        p[i] = (gt_y == i).sum().item() / p_y.size(0)
    ent = entropy(p)
    print('GT Entropy: %f' % ent)
    writer.add_scalar('selection_statistics/gt_entropy', ent, rd)

    # Border Entropy - 对大类别数用向量化加速
    if n_classes <= 100:
        # C=80 时 C*(C-1)/2 = 3160，可以接受
        c = idxs[:, 0:2].min(dim=1)[0] * 1000 + idxs[:, 0:2].max(dim=1)[0]
        n = int(n_classes * (n_classes - 1) / 2)
        p = np.zeros(n, dtype=float)
        idx = 0
        for i in range(n_classes - 1):
            for j in range(i + 1, n_classes):
                p[idx] = (c == (i * 1000) + j).sum().item() / p_y.size(0)
                idx += 1
        ent = entropy(p)
    else:
        # 类别数太多时跳过 border entropy
        ent = 0.0
    print('Border Entropy: %f' % ent)
    writer.add_scalar('selection_statistics/border_entropy', ent, rd)


def visualise_results(all_embeddings, can_idxs, Y_tr, q_idxs, path):
    if all_embeddings.shape[-1] == 2:
        tsne_results = all_embeddings
    else:
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(all_embeddings)

    lbls = Y_tr * 3
    lbls[can_idxs] += 1
    lbls[q_idxs] += 1
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.020, right=.995, top=.995, bottom=0.025, wspace=0.2, hspace=0.2)
    plt.axis('off')
    for label in range(Y_tr.max() + 1):
        c = list(mcolors.TABLEAU_COLORS.values())[label]
        ax.scatter(tsne_results[lbls == label * 3][:, 0],
                   tsne_results[lbls == label * 3][:, 1],
                   alpha=0.15, edgecolors='none', c=c, marker='o', s=30)
        ax.scatter(tsne_results[can_idxs][lbls[can_idxs] == label * 3 + 1][:, 0],
                   tsne_results[can_idxs][lbls[can_idxs] == label * 3 + 1][:, 1],
                   alpha=1., edgecolors='none', c=c, marker='*', s=180)
        ax.scatter(tsne_results[q_idxs][lbls[q_idxs] == label * 3 + 2][:, 0],
                   tsne_results[q_idxs][lbls[q_idxs] == label * 3 + 2][:, 1],
                   label=str(label), alpha=1., edgecolors='none', c=c, marker='o', s=120)
    ax.grid(False)
    plt.savefig(path, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General active learning hyper-parameters")

    parser.add_argument('--data_name', type=str, choices=['MNIST', 'EMNIST',
                                                          'SVHN', 'CIFAR10',
                                                          'CIFAR100', 'MiniImageNet',
                                                          'domain_net-real', 'mini_domain_net-real', 'tiny_domain_net-real',
                                                          'openml_6', 'openml_155', 'mirflickr', 'NUS-WIDE-TC21', 'MS-COCO'])
    parser.add_argument('--n_label', type=int, default=10, help='The number of distinct classes in the dataset.')

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_checkpoints', action='store_const', default=False, const=True)

    parser.add_argument('--save_images', action="store_const", default=False, const=True)
    parser.add_argument('--print_to_file', action="store_const", default=False, const=True)

    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 10, 100, 1000, 10000])
    parser.add_argument('--n_init_lb', type=int, default=100) #default=100
    parser.add_argument('--init_lb_method', type=str, default='general_random',
                        choices=['general_random', 'per_class_random', 'load'])
    parser.add_argument('--lb_init_path', type=str, default='')
    parser.add_argument('--n_query', type=int, default=100)
    parser.add_argument('--query_growth_ratio', type=int, default=1)
    parser.add_argument('--n_round', type=int, default=15)

    parser.add_argument('--strategy', type=str,
                        choices=['RandomSampling', 'EntropySampling',
                                 'BALDDropout', 'CoreSet',
                                 'AdversarialDeepFool',
                                 'BadgeSampling', 'CDALSampling', 'GCNSampling', 'AlphaMixSampling',
                                 'ContrastiveSampling', 'ContrastiveEntropySampling', 'UncertaintySampling',
                                 'All','VMFSampling','FullSupervised','MultiModalSampling','TypiClustSampling', 'ProbCoverSampling','CCMASampling', 'RLMBASampling',])


    #新增参数 ------------------------------------------------------------
    # ---------------- VMF Active Learning / CMR hyper-parameters ----------------
    # Query (sample selection)
    parser.add_argument('--lambda_density', type=float, default=0.5,
                        help='Diversity penalty strength in query: score = I_final - lambda_density * density')
    parser.add_argument('--lambda_bias', type=float, default=0.5,
                        help='Bias calibration strength in query/train')
    parser.add_argument('--alpha_sample', type=float, default=0.5,
                        help='Fusion weight for sample uncertainty: I = alpha*U_intra + (1-alpha)*U_inter')
    parser.add_argument('--kde_kappa', type=float, default=32.0,
                        help='vMF KDE kappa for density estimation')
    parser.add_argument('--kde_batch', type=int, default=512,
                        help='Chunk size for KDE density computation')

    # vMF intra uncertainty shape (IMPORTANT: currently not exposed unless we add these)
    parser.add_argument('--kappa_min', type=float, default=2.0,
                        help='Minimum kappa used in kappa mapping')
    parser.add_argument('--kappa_max', type=float, default=128.0,
                        help='Maximum kappa used in kappa mapping')
    parser.add_argument('--c_intra', type=float, default=0.08,
                        help='Slope for intra uncertainty: U_intra = 1 - sigmoid(c_intra * kappa)')

    # Train (sample weighting)
    parser.add_argument('--train_weight_alpha', type=float, default=1.0,
                        help='Weight coefficient for class uncertainty term')
    parser.add_argument('--train_weight_beta', type=float, default=1.0,
                        help='Weight coefficient for inter uncertainty term')
    parser.add_argument('--train_weight_gamma', type=float, default=1.0,
                        help='Weight coefficient for intra uncertainty term')
    parser.add_argument('--train_weight_temperature', type=float, default=1.0,
                        help='Temperature for training weights (power transform); 1.0 means no transform')

    # Label embedding update
    parser.add_argument('--label_momentum', type=float, default=0.9,
                        help='Momentum for label embedding update each AL round')

    parser.add_argument('--vmf_weight_warm_rounds', type=int, default=10,
                        help='Warm-up rounds for reweighting strength (scheme B)')

    #新增参数结束-----------------------------------------------------------------------------------------------------------

    parser.add_argument('--n_drop', type=int, default=5)
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--max_iter', type=int, default=50)

    # AlphaMix hyper-parameters
    parser.add_argument('--alpha_cap', type=float, default=0.03125)
    parser.add_argument('--alpha_opt', action="store_const", default=False, const=True)
    parser.add_argument('--alpha_closed_form_approx', action="store_const", default=False, const=True)

    # Gradient descent Alpha optimisation
    parser.add_argument('--alpha_learning_rate', type=float, default=0.1,
                        help='The learning rate of finding the optimised alpha')
    parser.add_argument('--alpha_clf_coef', type=float, default=1.0)
    parser.add_argument('--alpha_l2_coef', type=float, default=0.01)
    parser.add_argument('--alpha_learning_iters', type=int, default=5,
                        help='The number of iterations for learning alpha')
    parser.add_argument('--alpha_learn_batch_size', type=int, default=1000000)
    # ITR ==========================================config start==========================================
    parser.add_argument('--task', type=str, default='ITR')
    parser.add_argument('--cuda_visible_devices', type=str, default='0')
    parser.add_argument('--map_threshold', type=float, default=0.97)
    parser.add_argument('--K', type=int, default=7)

    #新增参数用于指定超参数组合进行训练
    parser.add_argument("--hyper_id", type=int, default=-1,
                        help=">=0: only run this hyper config id (no sweep)")
    parser.add_argument("--hyper_config_path", type=str, default="",
                        help="path to a json file; if set, run only this config")
    parser.add_argument("--disable_train_weight", action="store_true",
                        help="Disable vMF training weights (force uniform weights).")

    # 新增参数控制模块开关
    parser.add_argument('--enable_bias_calibration', action='store_true',
                        help="Enable bias calibration during sample selection")
    parser.add_argument('--enable_diversity_selection', action='store_true',
                        help="Enable diversity selection during sample selection")

    #是否关闭反馈机制
    parser.add_argument('--disable_kappa_calib', action='store_true',
                        help='Disable online kappa calibration (feedback loop).')
    # ===== 固定采样池支持（T1_fixed vs T2 单变量训练对比）=====
    # T2 运行时传空字符串（正常采样），T1_fixed 传入 T2 保存的目录（replay 模式）
    parser.add_argument('--fixed_query_dir', type=str, default='',
                        help='Dir to save/load per-round query indices. '
                             'If round_N.npy files exist → replay mode (skip AL query). '
                             'Otherwise normal AL, save indices after each round.')

    # ===== 静态消融支持 =====
    # 用途1：--save_final_lb_path   在 T2（完整框架）跑完后保存最终 idxs_lb 到 .npy 文件
    # 用途2：静态消融各变体通过 --init_lb_method load --lb_init_path <该文件> 加载，
    #         配合 --n_round 0 只做一次训练+评估，无 AL 循环
    parser.add_argument('--save_final_lb_path', type=str, default='',
                        help='If set, save final labeled boolean index (idxs_lb) to this .npy '
                             'after all AL rounds finish. Used for static ablation so all '
                             'variants train on exactly the same labeled set.')

    # ====== NEW (Scheme-2) vMF tuning knobs ======
    parser.add_argument('--kappa_soft_tau', type=float, default=0.07,
                        help='soft pooling temperature for unlabeled kappa (larger -> less winner-take-all)')
    parser.add_argument('--density_warmup_rounds', type=int, default=0,
                        help='density penalty warmup rounds; 0 disables warmup')
    parser.add_argument('--train_weight_scale', type=float, default=0.8,
                        help='centered training weight: w=1+scale*tanh(raw), range [1-scale,1+scale]')

    # ====== NEW: efficient hyper sweep (two-stage) ======
    parser.add_argument('--sweep_two_stage', action='store_true',
                        help='Enable two-stage hyper sweep (screen then full).')
    parser.add_argument('--sweep_screen_rounds', type=int, default=4,
                        help='Stage-1: AL rounds for screening (small budget).')
    parser.add_argument('--sweep_screen_epoch', type=int, default=2,
                        help='Stage-1: training epochs per round for screening.')
    parser.add_argument('--sweep_topk', type=int, default=10,
                        help='Stage-2: keep top-k configs from screening.')
    parser.add_argument('--sweep_score_metric', type=str, default='test_mAP',
                        help='metric column in round_metrics.csv to score configs, e.g., test_mAP')
    parser.add_argument('--sweep_score_mode', type=str, default='stable',
                        choices=['stable', 'last2', 'last'],
                        help='stable: mean-std-maxdrop; last2: mean of last2; last: last round')

    # os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_known_args()[0].cuda_visible_devices
    # print("xjz:"+str(os.environ["CUDA_VISIBLE_DEVICES"]))
    args, unknown = parser.parse_known_args()
    dataset = args.data_name
    #dataset = 'NUS-WIDE-TC21'
    # dataset = 'mirflickr'
    embedding = ''
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters
    # DATA_DIR = 'data/' + dataset + '/'
    DATA_DIR = parser.parse_known_args()[0].data_dir
    EVAL = False

    if dataset == 'mirflickr':
        parser.add_argument('--alpha', type=float, default=0.3)
        parser.add_argument('--beta', type=float, default=0.2)
        parser.add_argument('--max_epoch', type=int, default=40)
        parser.add_argument('--batch_size', type=int, default=100)
        parser.add_argument('--lr', type=float, default=5e-5)
        parser.add_argument('--betas', type=tuple, default=(0.5, 0.999))
        parser.add_argument('--t', type=float, default=0.4)
        parser.add_argument('--gnn', type=str, default='GCN')
        parser.add_argument('--n_layers', type=int, default=5)
        alpha = 0.3
        beta = 0.2
        max_epoch = 40
        batch_size = 100
        lr = 5e-5
        betas = (0.5, 0.999)
        t = 0.4
        gnn = 'GCN'
        n_layers = 5
    elif dataset == 'NUS-WIDE-TC21':
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--beta', type=float, default=0.2)
        parser.add_argument('--max_epoch', type=int, default=40)
        parser.add_argument('--batch_size', type=int, default=1024)
        parser.add_argument('--lr', type=float, default=5e-5)
        parser.add_argument('--betas', type=tuple, default=(0.5, 0.999))
        parser.add_argument('--t', type=float, default=0.3)
        parser.add_argument('--gnn', type=str, default='GAT')
        parser.add_argument('--n_layers', type=int, default=5)
        alpha = 0.2
        beta = 0.2
        max_epoch = 40
        batch_size = 1024
        lr = 5e-5
        betas = (0.5, 0.999)
        t = 0.3
        gnn = 'GAT'
        n_layers = 5
    elif dataset == 'MS-COCO':
        parser.add_argument('--alpha', type=float, default=2.8)
        parser.add_argument('--beta', type=float, default=0.2)
        parser.add_argument('--max_epoch', type=int, default=40)
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--lr', type=float, default=5e-5)
        parser.add_argument('--betas', type=tuple, default=(0.5, 0.999))
        parser.add_argument('--t', type=float, default=0.2)
        parser.add_argument('--gnn', type=str, default='GCN')
        parser.add_argument('--n_layers', type=int, default=5)
        alpha = 2.8
        beta = 0.2
        max_epoch = 40
        batch_size = 512
        lr = 5e-5
        betas = (0.5, 0.999)
        t = 0.2
        gnn = 'GCN'
        n_layers = 5
        k = 8
        gamma = 0.14
    else:
        raise NameError("Invalid dataset name!")
    print(f'...Dataset is {dataset}...')

    if embedding == 'glove':
        inp = loadmat("data/"+dataset+"/"+dataset + '-inp-glove6B.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'googlenews':
        inp = loadmat(dataset + '-inp-googlenews.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'fasttext':
        inp = loadmat(dataset + '-inp-fasttext.mat')['inp']
        inp = torch.FloatTensor(inp)
    else:
        inp = None
    print(f'...embedding is {embedding}...')
    args, _ = parser.parse_known_args()
    #添加一行逻辑，将其转换为 VMFSampling 可以识别的属性
    args.enable_kappa_calib = not args.disable_kappa_calib
    supervised_learning(args)
