import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision.transforms import v2


DATA_AUG = False

IMG = '/data/users/lkang/RVL-CDIP/images/'
LABEL = '/data/users/lkang/RVL-CDIP/RVL_CDIP_full.npy'


class RVL(Dataset):
    def __init__(self, img_dir, label_dir, split): # split: train, valid, test
        self.split = split
        data_all = np.load(label_dir, allow_pickle=True).item()
        self.data = data_all[split]
        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        self.img_proc_aug = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.RandomResizedCrop((224, 224), scale=(0.5, 1), ratio=(0.75, 1.25), antialias=True),
            v2.RandomAffine(degrees=5, shear=(-10, 10, -10, 10)),
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 5)),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        img_url = record[0]
        clas = record[1]
        gt = int(clas)

        img = Image.open(f'{IMG}{img_url}').convert('RGB')

        if self.split == 'train' and DATA_AUG:
            img_feat = self.img_proc_aug(img)
        else:
            img_feat = self.img_proc(img)

        sample_info = {'img_url': img_url,
                       'img': img_feat,
                       'label': gt,
                       }

        return sample_info


def loadData():
    data_dir = dict()
    for split in ['train', 'valid', 'test']:
        data_dir[split] = RVL(IMG, LABEL, split)
    return data_dir
    

if __name__ == '__main__':
    pass
