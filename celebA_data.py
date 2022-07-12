import torch
import os
import numpy as np
import pandas as pd

from skimage import io
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


"""
director structure should be constructed as below:

    img_align_celeba/
    ├── img_align_celeba
    ├── list_attr_celeba.csv
    ├── list_bbox_celeba.csv
    ├── list_eval_partition.csv
    └── list_landmarks_align_celeba.csv

usage:
    
    BATCH_SIZE = 64

    dataset = CelebA(
        root="data/img_align_celeba/", 
        csv_path="data/img_align_celeba/list_attr_celeba.csv",
        transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
"""

class CelebA(torch.utils.data.Dataset):
    def __init__(self, dset_root, csv_path, transforms=None):
        self.path = os.path.abspath(dset_root)
        self.label = pd.read_csv(os.path.abspath(csv_path))
        self.transforms = transforms

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.label.iloc[idx]["image_id"]
        img_path = os.path.abspath(os.path.join(self.path, img_id))
        img = io.imread(img_path)

        is_male = self.label.iloc[idx]["Male"]
        is_male = 1 if is_male == 1 else 0
        is_male = np.array(is_male)
        is_male = is_male.astype('long')

        if self.transforms:
            img = self.transforms(img)

        return img, is_male


def get_celeba_dataloader(args) -> tuple:
    """
        it returns (train_loader, val_loader) of celebA dset
    """

    celeba_dset = CelebA(
        args.dset_root, args.csv_root, 
        transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
    )

    train_ratio = args.split_ratio_train
    if not (0 <= train_ratio <= 1):
        raise RuntimeError("split ratio must be in range of [0, 1]")

    dset_size = len(celeba_dset)
    train_dset_size = int(dset_size * train_ratio)
    test_dset_size = dset_size - train_dset_size    

    train_dset, val_dset, _ = torch.utils.data.random_split(celeba_dset, [train_dset_size, test_dset_size, 0])
    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers        
    )
    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers        
    )

    return train_loader, val_loader