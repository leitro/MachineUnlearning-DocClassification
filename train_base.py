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
from dataset_all import loadData
from seed import set_seed
from model_resnet import Model
import torch.nn.functional as F
from util_log import save_model, LOG
import matplotlib.pyplot as plt


SCHE = False # scheduler

FIX_SEED = False
EARLY_STOP = 20

BATCH_SIZE = 1024
NUM_THREAD = 6
MAX_EPOCHS = 300
LEARNING_RATE = 2*1e-4
lr_milestone = [8, 16, 24, 32]
lr_gamma = 0.5

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def collate_batch(batch):
    new_batch = {k: [dic[k] for dic in batch] for k in batch[0]}
    new_batch['img'] = torch.tensor(np.array(new_batch['img'], dtype=np.float32))
    new_batch['label'] = torch.tensor(np.array(new_batch['label'], dtype=np.int64))
    return new_batch


def train(weights=None):
    if FIX_SEED:
        set_seed(42)
    data = loadData()
    dataloader_train = torch.utils.data.DataLoader(data['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREAD, collate_fn=collate_batch)
    dataloader_test = torch.utils.data.DataLoader(data['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, collate_fn=collate_batch)

    model = Model().to(DEVICE)

    if weights:
        checkpoint = torch.load(f'weights/{weights}')
        model.load_state_dict(checkpoint)
        print(f'Saved weights loaded: {weights}')

    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestone, gamma=lr_gamma)
    Evaluator = namedtuple('Evaluator', ['ce_loss', 'l2_loss'])
    evaluator = Evaluator(nn.CrossEntropyLoss(), nn.MSELoss())

    best_loss = 1e3
    best_acc = 0
    best_epoch = 0
    max_num = 0
    for epoch in range(1, MAX_EPOCHS+1):
        start_time = time.time()
        lr = scheduler.get_last_lr()[0]
        loss, acc = process_one_epoch(model, optimizer, evaluator, dataloader_train, 'train', epoch)
        print(f'#### TRAIN-{epoch} -- Loss[cla]: {loss:.2f}, Accuracy: {acc*100:.2f}%, lr: {lr:.6f}, time: {time.time()-start_time:.2f}s')
        if SCHE:
            scheduler.step()
        ## VALID
        loss_t, acc_t = process_one_epoch(model, None, evaluator, dataloader_test, 'valid', epoch)
        print(f'VALID(TEST)-{epoch} -- Loss[cla]: {loss_t:.2f}, Accuracy: {acc_t*100:.2f}%')

        if acc_t > best_acc:
            best_acc = acc_t
            best_epoch = epoch
            max_num = 0
        else:
            max_num += 1
        if max_num >= EARLY_STOP:
            print(f'[VALID] BEST ACC: {best_acc*100:.2f}%, BEST Epoch: {best_epoch}')
            ## TEST
            weights = f'weights/rvl-{best_epoch}.model'
            checkpoint = torch.load(weights)
            model.load_state_dict(checkpoint)
            loss_t, acc_t = process_one_epoch(model, None, evaluator, dataloader_test, 'test', best_epoch)
            print(f'TEST-{best_epoch} -- Loss[cla]: {loss_t:.2f}, Accuracy: {acc_t*100:.2f}%')
            return best_acc


def eval(best_epoch):
    data = loadData()
    Evaluator = namedtuple('Evaluator', ['ce_loss', 'l2_loss'])
    evaluator = Evaluator(nn.CrossEntropyLoss(), nn.MSELoss())
    weights = f'weights/rvl-{best_epoch}.model'
    model = Model().to(DEVICE)
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint)
    dataloader_test = torch.utils.data.DataLoader(data['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, collate_fn=collate_batch)
    loss_t, acc_t = process_one_epoch(model, None, evaluator, dataloader_test, 'test', best_epoch)
    print(f'TEST-{best_epoch} -- Loss[cla]: {loss_t:.2f}, Accuracy: {acc_t*100:.2f}%')

def rm_old_model(idx):
    models = glob.glob('weights/*.model')
    for m in models:
        epoch = int(m.split('-')[1].split('.')[0])
        if epoch < idx:
            os.system(f'rm weights/rvl-{epoch}.model')


def process_one_epoch(model, optimizer, evaluator, dataloader, split, epoch):
    total_loss = 0
    total_acc = 0
    count = 0
    cor_count = 0
    log = LOG(split, epoch)
    points = dict()
    res_acc = dict()
    for batch in tqdm(dataloader):
        images = batch['img'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        if split == 'train':
            feat512, res = model(images)
        else:
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

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        count += len(labels)
        cor_count += pred_count

        log.write(batch['img_url'], gts, preds)

    if split == 'train':
        save_model(model, epoch)

    acc_cada = dict()
    for key in res_acc.keys():
        acc_cada[key] = np.mean(res_acc[key]).item()

    acc_cada = list(acc_cada.items())
    acc_cada = sorted(acc_cada, key=lambda x: x[0])
    print('\n%: ', end='')
    for cla, acc in acc_cada:
        print(f'{cla}-{int(acc*100)}', end=' ')
    print('')
    return total_loss/count*BATCH_SIZE, cor_count/count


if __name__ == '__main__':
    os.system('rm imgs/*.jpg')
    os.system('rm logs/*.log')
    train()
