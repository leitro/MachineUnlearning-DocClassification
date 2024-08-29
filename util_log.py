import os
import torch

class LOG:
    def __init__(self, file_name, epoch):
        if not os.path.exists('logs'):
            os.makedirs('logs')  
        self.file = open(f'logs/{file_name}_epoch-{epoch}.log', 'a')
        self.file.write('Groundtruth | Prediction | Image URL\n')

    def write(self, img_urls, gts, preds):
        for img_url, gt, pred in list(zip(img_urls, gts, preds)):
            self.file.write(f'{gt} | {pred} | {img_url}\n')


def save_model(model, epoch, prefix=None):
    if not os.path.exists('weights'):
        os.makedirs('weights')
    if prefix:
        name = f'weights/unlearn-{prefix}-rvl-{epoch}.model'
    else:
        name = f'weights/rvl-{epoch}.model'
    torch.save(model.state_dict(), name)
