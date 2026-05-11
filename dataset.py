import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import openml
from sklearn.preprocessing import LabelEncoder
import pickle

from util import BackgroundGenerator

mini_domain_net_class_ids = {
    'bat': 20,
    'bathtub': 21,
    'bear': 23,
    'bed': 25,
    'bench': 28,
    'bicycle': 29,
    'bird': 31,
    'bus': 47,
    'butterfly': 49,
    'car': 61,
    'carrot': 62,
    'cat': 64,
    'chair': 68,
    'couch': 80,
    'cruise_ship': 86,
    'dog': 91,
    'pizza': 225,
    'strawberry': 290,
    'table': 301,
    'zebra': 343,
}


tiny_domain_net_class_ids = {
    'bat': 20,
    'bear': 23,
    'bicycle': 29,
    'bird': 31,
    'bus': 47,
    'butterfly': 49,
    'car': 61,
    'cat': 64,
    'dog': 91,
    'zebra': 343,
}


def is_openml(name):
    return len(name) > 6 and name[:6] == 'openml'


def get_openml_id(name):
    return int(name[7:])

# 核心数据加载入口
def get_dataset(name, data_dir):
    if name == 'MNIST':
        return get_MNIST(data_dir)
    if name == 'EMNIST':
        return get_EMNIST(data_dir)
    elif name == 'SVHN':
        return get_SVHN(data_dir)
    elif name == 'CIFAR10':
        return get_CIFAR10(data_dir)
    elif name == 'CIFAR100':
        return get_CIFAR100(data_dir)
    elif name == 'MiniImageNet':
        return get_MiniImageNet(data_dir)
    elif name == 'domain_net-real':
        return get_DomainNet_Real(data_dir)
    elif name == 'mini_domain_net-real':
        return get_Mini_DomainNet_Real(data_dir)
    elif name == 'tiny_domain_net-real':
        return get_Tiny_DomainNet_Real(data_dir)
    elif name == 'mirflickr' or name == 'NUS-WIDE-TC21' or name =='MS-COCO':
        return get_Mirflickr(data_dir)
    elif is_openml(name):
        return get_openml(data_dir, get_openml_id(name))


def get_MNIST(data_dir):
    raw_tr = datasets.MNIST(os.path.join(data_dir, 'MNIST'), train=True, download=True)
    raw_te = datasets.MNIST(os.path.join(data_dir, 'MNIST'), train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_te, Y_te


def get_EMNIST(data_dir):
    raw_tr = datasets.EMNIST(os.path.join(data_dir, 'EMNIST'), train=True, download=True, split='letters')
    raw_te = datasets.EMNIST(os.path.join(data_dir, 'EMNIST'), train=False, download=True, split='letters')
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets - 1
    X_te = raw_te.data
    Y_te = raw_te.targets - 1
    return X_tr, Y_tr, X_te, Y_te


def get_SVHN(data_dir):
    data_tr = datasets.SVHN(os.path.join(data_dir, 'SVHN'), split='train', download=True)
    data_te = datasets.SVHN(os.path.join(data_dir, 'SVHN'), split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te


def get_CIFAR10(data_dir):
    data_tr = datasets.CIFAR10(os.path.join(data_dir, 'CIFAR10'), train=True, download=True)
    data_te = datasets.CIFAR10(os.path.join(data_dir, 'CIFAR10'), train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te


def get_CIFAR100(data_dir):
    data_tr = datasets.CIFAR100(os.path.join(data_dir, 'CIFAR100'), train=True, download=True)
    data_te = datasets.CIFAR100(os.path.join(data_dir, 'CIFAR100'), train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te


def get_MiniImageNet(data_dir):
    f = open(os.path.join(data_dir, 'MiniImageNet', 'mini-imagenet-cache-train.pkl'), 'rb')
    train_data = pickle.load(f)
    f = open(os.path.join(data_dir, 'MiniImageNet', 'mini-imagenet-cache-val.pkl'), 'rb')
    val_data = pickle.load(f)
    f = open(os.path.join(data_dir, 'MiniImageNet', 'mini-imagenet-cache-test.pkl'), 'rb')
    test_data = pickle.load(f)

    labels = list(train_data['class_dict'].keys()) + list(val_data['class_dict'].keys()) + list(test_data['class_dict'].keys())
    image_count = len(train_data['class_dict'][labels[0]])
    test_proportion = int(image_count * 0.2)
    train_proportion = image_count - test_proportion

    image_dim = train_data['image_data'].shape[1]
    X_tr = np.zeros((len(labels) * train_proportion, image_dim, image_dim, 3), dtype=np.uint8)
    Y_tr = torch.ones((len(labels) * train_proportion), dtype=torch.long)
    X_te = np.zeros((len(labels) * test_proportion, image_dim, image_dim, 3), dtype=np.uint8)
    Y_te = torch.ones((len(labels) * test_proportion), dtype=torch.long)

    idx = 0
    for label in train_data['class_dict']:
        X_te[idx * test_proportion:(idx + 1) * test_proportion] = train_data['image_data'][train_data['class_dict'][label][:test_proportion]]
        Y_te[idx * test_proportion:(idx + 1) * test_proportion] *= labels.index(label)

        X_tr[idx * train_proportion:(idx + 1) * train_proportion] = train_data['image_data'][train_data['class_dict'][label][test_proportion:]]
        Y_tr[idx * train_proportion:(idx + 1) * train_proportion] *= labels.index(label)

        idx += 1

    for label in val_data['class_dict']:
        X_te[idx * test_proportion:(idx + 1) * test_proportion] = val_data['image_data'][val_data['class_dict'][label][:test_proportion]]
        Y_te[idx * test_proportion:(idx + 1) * test_proportion] *= labels.index(label)

        X_tr[idx * train_proportion:(idx + 1) * train_proportion] = val_data['image_data'][val_data['class_dict'][label][test_proportion:]]
        Y_tr[idx * train_proportion:(idx + 1) * train_proportion] *= labels.index(label)

        idx += 1

    for label in test_data['class_dict']:
        X_te[idx * test_proportion:(idx + 1) * test_proportion] = test_data['image_data'][test_data['class_dict'][label][:test_proportion]]
        Y_te[idx * test_proportion:(idx + 1) * test_proportion] *= labels.index(label)

        X_tr[idx * train_proportion:(idx + 1) * train_proportion] = test_data['image_data'][test_data['class_dict'][label][test_proportion:]]
        Y_tr[idx * train_proportion:(idx + 1) * train_proportion] *= labels.index(label)

        idx += 1

    return X_tr, Y_tr, X_te, Y_te


def get_DomainNet_Real(data_dir):
    data_dir = os.path.join(data_dir, 'domain_net-real')

    X_tr, Y_tr, X_te, Y_te = [], [], [], []

    with open(os.path.join(data_dir, 'real_train.txt'), 'r') as f:
        for item in f.readlines():
            feilds = item.strip()
            name, label = feilds.split(' ')
            X_tr.append(os.path.join(data_dir, name))
            Y_tr.append(int(label))

    with open(os.path.join(data_dir, 'real_test.txt'), 'r') as f:
        for item in f.readlines():
            feilds = item.strip()
            name, label = feilds.split(' ')
            X_te.append(os.path.join(data_dir, name))
            Y_te.append(int(label))

    return np.array(X_tr), torch.from_numpy(np.array(Y_tr)), np.array(X_te), torch.from_numpy(np.array(Y_te))


def get_Mini_DomainNet_Real(data_dir):
    data_dir = os.path.join(data_dir, 'domain_net-real')

    X_tr, Y_tr, X_te, Y_te = [], [], [], []
    label_map = {}

    with open(os.path.join(data_dir, 'real_train.txt'), 'r') as f:
        for item in f.readlines():
            feilds = item.strip()
            name, label = feilds.split(' ')
            label = int(label)
            if label in mini_domain_net_class_ids.values():
                X_tr.append(os.path.join(data_dir, name))
                if label not in label_map:
                    label_map[label] = len(label_map)
                Y_tr.append(label_map[label])

    with open(os.path.join(data_dir, 'real_test.txt'), 'r') as f:
        for item in f.readlines():
            feilds = item.strip()
            name, label = feilds.split(' ')
            label = int(label)
            if label in mini_domain_net_class_ids.values():
                X_te.append(os.path.join(data_dir, name))
                if label not in label_map:
                    label_map[label] = len(label_map)
                Y_te.append(label_map[label])

    return np.array(X_tr), torch.from_numpy(np.array(Y_tr)), np.array(X_te), torch.from_numpy(np.array(Y_te))


def get_Tiny_DomainNet_Real(data_dir):
    data_dir = os.path.join(data_dir, 'domain_net-real')

    X_tr, Y_tr, X_te, Y_te = [], [], [], []
    label_map = {}

    with open(os.path.join(data_dir, 'real_train.txt'), 'r') as f:
        for item in f.readlines():
            feilds = item.strip()
            name, label = feilds.split(' ')
            label = int(label)
            if label in tiny_domain_net_class_ids.values():
                X_tr.append(os.path.join(data_dir, name))
                if label not in label_map:
                    label_map[label] = len(label_map)
                Y_tr.append(label_map[label])

    with open(os.path.join(data_dir, 'real_test.txt'), 'r') as f:
        for item in f.readlines():
            feilds = item.strip()
            name, label = feilds.split(' ')
            label = int(label)
            if label in tiny_domain_net_class_ids.values():
                X_te.append(os.path.join(data_dir, name))
                if label not in label_map:
                    label_map[label] = len(label_map)
                Y_te.append(label_map[label])

    return np.array(X_tr), torch.from_numpy(np.array(Y_tr)), np.array(X_te), torch.from_numpy(np.array(Y_te))


def get_openml(data_dir, dataset_id):
    openml.config.apikey = '3411e20aff621cc890bf403f104ac4bc'
    openml.config.set_cache_directory(data_dir)
    ds = openml.datasets.get_dataset(dataset_id)
    data = ds.get_data(target=ds.default_target_attribute)
    X = np.asarray(data[0], dtype=np.int64)
    y = np.asarray(data[1])
    y = LabelEncoder().fit(y).transform(y)

    nClasses = int(max(y) + 1)
    nSamps, dim = np.shape(X)
    testSplit = .1
    inds = np.random.permutation(nSamps)
    X = X[inds]
    y = y[inds]

    split = int((1. - testSplit) * nSamps)
    while True:
        inds = np.random.permutation(split)
        if len(inds) > 50000: inds = inds[:50000]
        X_tr = X[:split]
        X_tr = X_tr[inds]
        X_tr = torch.Tensor(X_tr)

        y_tr = y[:split]
        y_tr = y_tr[inds]
        Y_tr = torch.Tensor(y_tr).long()

        X_te = torch.Tensor(X[split:])
        Y_te = torch.Tensor(y[split:]).long()

        if len(np.unique(Y_tr)) == nClasses: break

    return X_tr, Y_tr, X_te, Y_te


# 数据处理类（Dataset 封装）
def get_handler(name):
    if name == 'MNIST':
        return DataHandler1
    if name == 'EMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3
    elif name == 'CIFAR100':
        return DataHandler3
    elif name == 'MiniImageNet':
        return DataHandler3
    elif name == 'domain_net-real':
        return DataHandler4
    elif name == 'mini_domain_net-real':
        return DataHandler4
    elif name == 'tiny_domain_net-real':
        return DataHandler4
    elif name == 'mirflickr' or name == 'NUS-WIDE-TC21' or name == 'MS-COCO':
        return CustomDataSet
    elif is_openml(name):
        return DataHandler5


class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, idx):
        x = self.transform(Image.open(self.X[idx]))
        class_id = self.Y[idx]
        y = class_id.clone().detach()
        return x, y, idx

    def __len__(self):
        return len(self.X)


class DataHandler5(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)

from scipy.io import loadmat

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# 返回 图像特征 + 文本特征 + 标签 + 索引
class CustomDataSet(torch.utils.data.dataset.Dataset):
    def __init__(
            self,
            images,
            texts,
            labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label, index

    def __len__(self):
        count = len(self.images)
        return count


"""
加载训练和测试特征
创建 CustomDataSet
包装成 DataLoaderX
返回：
    dataloader：训练和测试 DataLoader
    input_data_par：原始特征、维度、类别数，用于模型初始化
"""
def get_loader(path, batch_size):
    img_train = loadmat(path + "train_img.mat")['train_img']    # trn_img_num(18013), 4096 fastrcnn
    img_test = loadmat(path + "test_img.mat")['test_img']   # tst_img_num(2002), 4096
    text_train = loadmat(path + "train_txt.mat")['train_txt']   # trn_txt_num(18013), 300  # glove
    text_test = loadmat(path + "test_txt.mat")['test_txt']      # 2002, 300
    label_train = loadmat(path + "train_lab.mat")['train_lab']  # trn_num(18013), 21
    label_test = loadmat(path + "test_lab.mat")['test_lab']     # 2002, 24

    imgs = {'train': img_train, 'test': img_test}
    texts = {'train': text_train, 'test': text_test}
    labels = {'train': label_train, 'test': label_test}
    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['train', 'test']}

    shuffle = {'train': True, 'test': False}

    dataloader = {x: DataLoaderX(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    img_dim = img_train.shape[1]
    text_dim = text_train.shape[1]
    num_class = label_train.shape[1]

    input_data_par = {}
    input_data_par['img_test'] = img_test
    input_data_par['text_test'] = text_test
    input_data_par['label_test'] = label_test
    input_data_par['img_train'] = img_train
    input_data_par['text_train'] = text_train
    input_data_par['label_train'] = label_train
    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    return dataloader, input_data_par


# 这些是跨模态检索特征（已经提取好，不是原始图片）
# def get_Mirflickr(path):
#     img_train = loadmat(path + "train_img.mat")['train_img']  # trn_img_num(18013), 4096 fastrcnn
#     img_test = loadmat(path + "test_img.mat")['test_img']  # tst_img_num(2002), 4096
#     text_train = loadmat(path + "train_txt.mat")['train_txt']  # trn_txt_num(18013), 300  # glove
#     text_test = loadmat(path + "test_txt.mat")['test_txt']  # 2002, 300
#     label_train = loadmat(path + "train_lab.mat")['train_lab']  # trn_num(18013), 21
#     label_test = loadmat(path + "test_lab.mat")['test_lab']  # 2002, 24
#
#     return img_train, text_train, label_train, img_test, text_test, label_test


def _load_pkl(p):
    with open(p, "rb") as f:
        obj = pickle.load(f)
    return obj

# def get_Mirflickr(path):
#     """
#     用 CLIP 特征的 PKL 文件替换原 .mat 读取方式
#     期望目录下有：
#       - train_25k.pkl  dict: image/text/label
#       - query_25k.pkl  dict: image/text/label
#       - eval_25k.pkl   dict: image/text/label  (可选)
#       - label_prompt_features.pkl  np.ndarray [24,512] (可选，给你的语义锚点策略用)
#     """
#     train_p = os.path.join(path, "eval_25k.pkl")
#     query_p = os.path.join(path, "query_25k.pkl")
#
#     train = _load_pkl(train_p)
#     query = _load_pkl(query_p)
#
#     img_train = np.asarray(train["image"], dtype=np.float32)  # [Ntr,512]
#     txt_train = np.asarray(train["text"], dtype=np.float32)   # [Ntr,512]
#     lab_train = np.asarray(train["label"], dtype=np.float32)  # [Ntr,24]
#
#     img_test = np.asarray(query["image"], dtype=np.float32)   # [Nte,512]
#     txt_test = np.asarray(query["text"], dtype=np.float32)    # [Nte,512]
#     lab_test = np.asarray(query["label"], dtype=np.float32)   # [Nte,24]
#
#     return img_train, txt_train, lab_train, img_test, txt_test, lab_test

def get_Mirflickr(path):
    """
    根据 path 最后一层目录名，自动适配不同数据集的 PKL 命名规则
    支持：
      - flickr25k
      - nus-wide-21
      - coco / ms-coco

    每个 pkl 期望是 dict，包含：
      image / text / label
    """

    dirname = os.path.basename(os.path.normpath(path)).lower()

    # ---------- 1. 根据目录名确定文件名 ----------
    if dirname in ["flickr25k", "mirflickr", "mirflickr25k","25k"]:
        train_p = os.path.join(path, "eval_25k.pkl")
        query_p = os.path.join(path, "query_25k.pkl")

    elif dirname in ["nus-wide-21", "nuswide-21", "nus-wide"]:
        train_p = os.path.join(path, "retrival.pkl")
        query_p = os.path.join(path, "query.pkl")

    elif dirname in ["coco", "ms-coco", "mscoco"]:
        train_p = os.path.join(path, "eval_coco.pkl")
        query_p = os.path.join(path, "query_coco.pkl")

    else:
        raise ValueError(
            f"Unknown dataset directory: {path}\n"
            f"Expected directory name like flickr25k / nus-wide-21 / coco"
        )

    # ---------- 2. 检查文件是否存在 ----------
    if not os.path.exists(train_p):
        raise FileNotFoundError(f"Train file not found: {train_p}")
    if not os.path.exists(query_p):
        raise FileNotFoundError(f"Query file not found: {query_p}")

    # ---------- 3. 读取数据 ----------
    train = _load_pkl(train_p)
    query = _load_pkl(query_p)

    img_train = np.asarray(train["image"], dtype=np.float32)
    txt_train = np.asarray(train["text"], dtype=np.float32)
    lab_train = np.asarray(train["label"], dtype=np.float32)

    img_test = np.asarray(query["image"], dtype=np.float32)
    txt_test = np.asarray(query["text"], dtype=np.float32)
    lab_test = np.asarray(query["label"], dtype=np.float32)

    if dirname in ["nus-wide-21", "nuswide-21", "nus-wide"]:
        n_total = img_test.shape[0]

        # 只在 query 恰好是 4200（或大于 2000）时做切分；否则保持原样
        target_test = 2000
        if n_total > target_test:
            n_move = n_total - target_test  # 4200 -> move 2200

            # 固定随机种子，保证每次切分一致（可复现）
            rng = np.random.default_rng(0)
            perm = rng.permutation(n_total)
            move_idx = perm[:n_move]
            keep_idx = perm[n_move:]

            # move 到 train（扩大 pool）
            img_train = np.concatenate([img_train, img_test[move_idx]], axis=0)
            txt_train = np.concatenate([txt_train, txt_test[move_idx]], axis=0)
            lab_train = np.concatenate([lab_train, lab_test[move_idx]], axis=0)

            # keep 作为最终 test（2000）
            img_test = img_test[keep_idx]
            txt_test = txt_test[keep_idx]
            lab_test = lab_test[keep_idx]

            # 可选：打印确认（你不想打印可以删掉）
            print(
                f"[NUS-WIDE split] moved {n_move} from query->train; test={img_test.shape[0]}, train={img_train.shape[0]}")

    return img_train, txt_train, lab_train, img_test, txt_test, lab_test