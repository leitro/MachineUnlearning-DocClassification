import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision.transforms import v2


IMG = '/data/users/lkang/RVL-CDIP/images/'
LABEL_F = '/data/users/lkang/RVL-CDIP/RVL_CDIP_full.npy'

LABEL_S = 'RVL_CDIP_train_class_distil_0.1_dict_mix.npy'

NUM_CLASS = 16

UNCLASS = [0] # unlearn classes


class RVL(Dataset):
    '''retain: True return retain set, False return unlearn set'''
    def __init__(self, img_dir, label_dir, unlearn_classes, split, retain_o_forget, random_label=False): # split: train, valid, test
        self.random_label = random_label
        data_all = np.load(label_dir, allow_pickle=True).item()
        self.data = self._retain_unlearn(data_all[split], unlearn_classes, retain_o_forget)
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
            v2.RandomResizedCrop((224, 224), scale=(0.7, 1), ratio=(0.85, 1.15), antialias=True),
            v2.RandomHorizontalFlip(p=0.4),
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 5)),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.4),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def _retain_unlearn(self, data, unlearn_classes, retain_o_forget): # return: retain_set or forget_set
        if retain_o_forget: # True: retain_set
            return [item for item in data if int(item[1]) not in unlearn_classes]
        else: # False: forget_set
            return [item for item in data if int(item[1]) in unlearn_classes]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        img_url = record[0]
        if self.random_label:
            ids = list(range(NUM_CLASS))
            for cla in UNCLASS:
                ids.remove(cla)
            gt = random.choice(ids)
        else:
            clas = record[1]
            gt = int(clas)

        img = Image.open(f'{IMG}{img_url}').convert('RGB')
        img_feat = self.img_proc(img)


        sample_info = {'img_url': img_url,
                       'img': img_feat,
                       'label': gt,
                       }

        return sample_info

class RVLGEN(Dataset):
    '''retain: True return retain set, False return unlearn set'''
    def __init__(self, img_dir, label_dir, unlearn_classes, split, retain_o_forget, random_label=False): # split: train, valid, test
        self.random_label = random_label
        data_all = np.load(label_dir, allow_pickle=True).item()
        self.data = self._retain_unlearn(data_all[split], unlearn_classes, retain_o_forget)
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
            v2.RandomResizedCrop((224, 224), scale=(0.7, 1), ratio=(0.85, 1.15), antialias=True),
            v2.RandomHorizontalFlip(p=0.4),
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 5)),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.4),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def _retain_unlearn(self, data, unlearn_classes, retain_o_forget): # return: retain_set or forget_set
        if retain_o_forget: # True: retain_set
            return [item for item in data if int(item[1]) not in unlearn_classes]
        else: # False: forget_set
            return [item for item in data if int(item[1]) in unlearn_classes]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        img_url = record[0]
        if self.random_label:
            ids = list(range(NUM_CLASS))
            for cla in UNCLASS:
                ids.remove(cla)
            gt = random.choice(ids)
        else:
            clas = record[1]
            gt = int(clas)

        if gt in UNCLASS:
            img_feat = np.random.rand(3, 224, 224)
        else:
            img = Image.open(f'{IMG}{img_url}').convert('RGB')
            img_feat = self.img_proc(img)


        sample_info = {'img_url': img_url,
                       'img': img_feat,
                       'label': gt,
                       }

        return sample_info



def loadData_full():
    data_dir = dict()
    for split in ['train', 'valid', 'test']:
        data_dir[split] = dict()
        data_dir[split]['retain'] = RVL(IMG, LABEL_F, UNCLASS, split, True)
        data_dir[split]['forget'] = RVL(IMG, LABEL_F, UNCLASS, split, False)
    return data_dir

def loadData_randGen():
    split = 'train'
    data_dir = dict()
    data_dir[split] = dict()
    data_dir[split]['retain'] = RVLGEN(IMG, LABEL_S, UNCLASS, split, True)
    data_dir[split]['forget'] = RVLGEN(IMG, LABEL_S, UNCLASS, split, False)
    data_dir[split]['forget_randomlabel'] = RVLGEN(IMG, LABEL_S, UNCLASS, split, False, True)
    return data_dir


if __name__ == '__main__':
    res = loadData_subset_contour('RVL_CDIP_train_class_distil_0.1_dict_mix.npy')
    next(iter(res['train']['forget']))

