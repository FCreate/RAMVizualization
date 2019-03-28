from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from skimage import io, transform
import numpy as np
from utils import plot_images
from config import Config
import os
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import Subset

import os

class ToxicDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        print (csv_file)
        self.data = pd.read_csv(csv_file)
        self.max_tox=self.data.loc[:, (self.data.columns != 'SMILES')& (self.data.columns !='Unnamed: 0')].as_matrix()
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.max_tox)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,str(idx)+'.png')
        image = io.imread(img_name)
        image = image[np.newaxis, :, :]
        image.astype(float)
        y = self.max_tox[idx]
        sample = {'image': image, 'y': y}
        if self.transform:
            sample = self.transform
        return sample['image'], sample['y']
    
def get_train_valid_loader(dataset,
                           indices,
                           data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=False,
                           show_sample=False,
                           cv = False,
                           num_workers=4,
                           pin_memory=True,fold=0):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MNIST dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
      In the paper, this number is set to 0.1.
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    #normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor()#, normalize,
    ])

    # load dataset
    dataset = ToxicDataset(csv_file="aggregate_tox.csv", root_dir=Config().data_dir)
    num_train = len(dataset)
    indices_all = list(range(num_train))
    valid_size = 1.0/5.0
    valid_size_nums = int(np.floor(valid_size*num_train))
    val_indices = list(np.random.choice(indices, valid_size_nums))
    val_indices_return = list(set(indices)-set(val_indices))
    train_indices = list(set(indices_all)-set(val_indices))
    #start_split = int(np.floor(float(fold/5.0)*num_train))
    #stop_split = int(np.floor(float(fold/5.0)*num_train + valid_size*num_train))
    #split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
    #train_idx, valid_idx = indices[:start_split]+indices[stop_split:], indices[start_split:stop_split]
    train_idx, valid_idx = train_indices, val_indices
    
    #train_dataset = Subset(dataset = dataset, indices = train_idx)
    #valid_dataset = Subset(dataset = dataset, indices = valid_idx)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    #train_sampler = SequenttialSampler(train_idx)
    #valid_sampler = SequentialSampler(valid_idx)

    '''train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )'''
    
    '''train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )'''
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,sampler=train_sampler, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,sampler = valid_sampler, pin_memory=pin_memory)

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return ((train_loader, valid_loader), val_indices_return)


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    # load dataset
    dataset = ToxicDataset(csv_file="aggregate_tox.csv", root_dir=Config().data_dir)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


