from collections import namedtuple
from transformers import ResNetModel
import glob
import cv2
import os
import numpy as np
import time
import random
from tqdm import tqdm
import torch
from torch import nn
from dataset_unlearn_aug import loadData_full, loadData_subset, loadData_subset_contour, NUM_CLASS, UNCLASS
from seed import set_seed
from model_resnet import Model
import torch.nn.functional as F
from util_log import save_model, LOG


FIX_SEED = False

BATCH_SIZE = 1024
NUM_THREAD = 6
MAX_ITER = 500
ITER = 100
LEARNING_RATE = 2*1e-4

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def collate_batch(batch):
    new_batch = {k: [dic[k] for dic in batch] for k in batch[0]}
    new_batch['img'] = torch.tensor(np.array(new_batch['img'], dtype=np.float32))
    new_batch['label'] = torch.tensor(np.array(new_batch['label'], dtype=np.int64))
    return new_batch


def train(prefix, label_sc, weights=None):
    if FIX_SEED:
        set_seed(42)
    data = loadData_subset_contour(label_sc)
    dataloader_train_r = torch.utils.data.DataLoader(data['train']['retain'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREAD, collate_fn=collate_batch)
    dataloader_train_f = torch.utils.data.DataLoader(data['train']['forget'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREAD, collate_fn=collate_batch)
    if prefix == 'randomlabel':
        data_cat_randomlabel = torch.utils.data.ConcatDataset([data['train']['retain'], data['train']['forget_randomlabel']])
        dataloader_cat_randomlabel = torch.utils.data.DataLoader(data_cat_randomlabel, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREAD, collate_fn=collate_batch)
        dataloader_cur = dataloader_cat_randomlabel
    elif prefix == 'finetune':
        dataloader_cur = dataloader_train_r
    elif prefix == 'retrain':
        dataloader_cur = dataloader_train_r
    else:
        print('Available prefix: ["retrain", "finetune", "randomlabel"]')
        exit()

    model = Model().to(DEVICE)

    if weights:
        checkpoint = torch.load(f'weights/{weights}')
        model.load_state_dict(checkpoint)
        print(f'Saved weights loaded: {weights}')

    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    Evaluator = namedtuple('Evaluator', ['ce_loss', 'l2_loss'])
    evaluator = Evaluator(nn.CrossEntropyLoss(), nn.MSELoss())

    total_loss = []
    n_iter = 0
    start_time = time.time()
    fin_acc = dict()
    while n_iter < MAX_ITER:
        for batch in dataloader_cur:
            n_iter += 1
            images = batch['img'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            feat512, res = model(images)
            loss = evaluator.ce_loss(res, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())

            if n_iter % ITER == 0:
                save_model(model, n_iter, prefix)

                print(f'[{prefix}] TRAIN-ITER:{n_iter} -- Loss[cla]: {np.mean(total_loss):.2f}, time: {time.time()-start_time:.2f}s')
                start_time = time.time()
                total_loss = []
                ## VALID
                acc_r, acc_f = eval(n_iter, 'train', prefix)
                acc_rt, acc_ft = eval(n_iter, 'test', prefix)
                fin_acc[f'iter-{str(n_iter)}'] = (acc_r, acc_f, acc_rt, acc_ft)

    return fin_acc



def eval(best_epoch, split, prefix):
    data = loadData_full()
    Evaluator = namedtuple('Evaluator', ['ce_loss', 'l2_loss'])
    evaluator = Evaluator(nn.CrossEntropyLoss(), nn.MSELoss())
    weights = f'weights/unlearn-{prefix}-rvl-{best_epoch}.model'
    print(f'loading {weights}')
    model = Model().to(DEVICE)
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint)
    dataloader = torch.utils.data.DataLoader(data[split]['retain'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, collate_fn=collate_batch)
    loss_r, acc_r = eval_one_epoch(model, evaluator, dataloader, best_epoch)
    print(f'{split}[retain]-{best_epoch} -- Loss[cla]: {loss_r:.2f}, Accuracy: {acc_r*100:.2f}%')
    dataloader = torch.utils.data.DataLoader(data[split]['forget'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, collate_fn=collate_batch)
    loss_f, acc_f = eval_one_epoch(model, evaluator, dataloader, best_epoch)
    print(f'{split}[forget]-{best_epoch} -- Loss[cla]: {loss_f:.2f}, Accuracy: {acc_f*100:.2f}%')
    return acc_r, acc_f


def eval_one_epoch(model, evaluator, dataloader, epoch):
    total_loss = 0
    total_acc = 0
    count = 0
    cor_count = 0
    points = dict()
    res_acc = dict()
    for batch in dataloader:
        images = batch['img'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        with torch.no_grad():
            feat512, res = model(images)

        loss = evaluator.ce_loss(res, labels)
        preds = res.argmax(-1).detach().cpu().numpy()
        gts = labels.detach().cpu().numpy()
        for pred, gt in zip(preds, gts):
            if gt in res_acc:
                res_acc[gt].append(1 if pred==gt else 0)
            else:
                res_acc[gt] = [1 if pred==gt else 0]

        pred_count = np.sum(preds == gts)

        total_loss += loss.item()
        count += len(labels)
        cor_count += pred_count

    acc_cada = dict()
    for key in res_acc.keys():
        acc_cada[key] = np.mean(res_acc[key]).item()

    acc_cada = list(acc_cada.items())
    acc_cada = sorted(acc_cada, key=lambda x: x[0])
    return total_loss/count*BATCH_SIZE, cor_count/count

def mean_std(item_list): # [(acc_r, acc_f, acc_rt, acc_ft), ...] num of redo times
    acc_r = [item[0] for item in item_list]
    acc_f = [item[1] for item in item_list]
    acc_rt = [item[2] for item in item_list] 
    acc_ft = [item[3] for item in item_list] 
    print(f'acc_r: {np.mean(acc_r):.5f}+-{np.std(acc_r):.5f}')
    print(f'acc_f: {np.mean(acc_f):.5f}+-{np.std(acc_f):.5f}')
    print(f'acc_rt: {np.mean(acc_rt):.5f}+-{np.std(acc_rt):.5f}') 
    print(f'acc_ft: {np.mean(acc_ft):.5f}+-{np.std(acc_ft):.5f}')


if __name__ == '__main__':
    '''
    [
    [{'0.01-random': [{'iter-100': (acc_r, acc_f, acc_rt, acc_ft), ...}, #rt
                   {'iter-100': (acc_r, acc_f, acc_rt, acc_ft), ...}, #ft
                   {'iter-100': (acc_r, acc_f, acc_rt, acc_ft), ...}] #rl
     }]
    ] # number of exps 3
    '''

    for ii in range(4):
        result = dict()
        for ratio in [0.1, 0.05, 0.01]:
            for tb in ['random', 'top', 'bottom', 'mix']:
                key = f'{ratio}-{tb}'
                result[key] = []
                label_sc = f'LABEL_SC/RVL_CDIP_train_class_distil_{ratio}_dict_{tb}.npy'
                print(label_sc)
                ret = train('retrain', label_sc)
                result[key].append(ret)
                ret = train('finetune', label_sc, 'rvl-41.model.87.93.scratch')
                result[key].append(ret)
                ret = train('randomlabel', label_sc, 'rvl-41.model.87.93.scratch')
                result[key].append(ret)

        np.save(f'result_iter100-500_trial{ii}.npy', result)

