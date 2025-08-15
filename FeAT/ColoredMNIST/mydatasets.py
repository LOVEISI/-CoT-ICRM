
import numpy as np
import torch
from torchvision import datasets
import math
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate


from misc import split_dataset,make_weights_for_balanced_classes,seed_hash
# from fast_data_loader import InfiniteDataLoader, FastDataLoader


#coloredmnist are modified from https://github.com/facebookresearch/InvariantRiskMinimization
def coloredmnist(label_noise_rate, trenv1, trenv2, int_target=False):
    # Load MNIST, make train/val splits, and shuffle train set examples
    mnist = datasets.MNIST('../data/MNIST', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])
    
    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Build environments
    def make_environment(images, labels, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(label_noise_rate, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
        
        if int_target:
            return {
                'images': (images.float() / 255.), 
                'labels': labels[:, None].long().flatten()
            }
        else:
            return {
                'images': (images.float() / 255.), 
                'labels': labels[:, None]
            }
             

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], trenv1),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], trenv2),
    ]
    # init 3 test environments [0.1, 0.5, 0.9] 
    test_envs = [    
        make_environment(mnist_val[0], mnist_val[1], 0.9),
        make_environment(mnist_val[0], mnist_val[1], 0.1),
        make_environment(mnist_val[0], mnist_val[1], 0.5),
    ]
    return envs, test_envs

# mydatasets.py
import os, json
import numpy as np
import torch

def femnist(data_dir, int_target=False):
    """
    加载 FEMNIST 数据集：
    - data_dir 下有 train/ 和 test/ 两个子目录
    - 每个子目录下若干 all_data_*.json 文件，格式为:
        {
          "users": [...],
          "num_samples": [...],
          "user_data": {
             "f3449_11": {"x": [[...],...], "y": [..]},
             ...
          }
        }
    返回 (envs, test_envs)，每个 env 是 {'images': Tensor, 'labels': Tensor}
    """
    def load_env(path):
        with open(path, 'r') as f:
            data = json.load(f)

        # data['users'] 是一个用户列表，我们把所有用户的数据 concat 到一起
        all_x = []
        all_y = []
        for u in data['users']:
            ux = np.array(data['user_data'][u]['x'], dtype=np.float32)
            uy = np.array(data['user_data'][u]['y'], dtype=np.int64)
            all_x.append(ux)
            all_y.append(uy)
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        # 如果是扁平 784 维，就 reshape 成 (N,28,28)
        if all_x.ndim == 2 and all_x.shape[1] == 784:
            all_x = all_x.reshape(-1, 28, 28)
        # 转为 (N,1,28,28) 并归一化到 [0,1]
        imgs = torch.from_numpy(all_x).unsqueeze(1) / 255.0
        if int_target:
            labels = torch.from_numpy(all_y).flatten()       # (N,)
        else:
            # labels = torch.from_numpy(all_y).astype(torch.float32).unsqueeze(1)  # (N,1)
            labels = torch.from_numpy(all_y).float().unsqueeze(1)  # (N,1)
        return {'images': imgs, 'labels': labels}

    # 遍历 train/ 和 test/ 目录
    def load_envs(subdir):
        root = os.path.join(data_dir, subdir)
        envs = []
        for fname in sorted(os.listdir(root)):
            if fname.endswith('.json'):
                envs.append(load_env(os.path.join(root, fname)))
        return envs

    train_envs = load_envs('train')
    test_envs  = load_envs('test')
    return train_envs, test_envs
