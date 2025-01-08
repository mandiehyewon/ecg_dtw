import os
import sys
import argparse
from tqdm import tqdm
import numpy as np

import torch

from config import args
from data import get_dtwdata, get_data_iterator
from model import get_model
from utils.loss import get_loss
from utils.metrics import Evaluator
from utils.utils import set_devices, calc_dtw_lead
from utils.soft_dtw_cuda import SoftDTW

'''
CUDA_VISIBLE_DEVICES=1 python test_dtwsurrogate.py --ecg-len 1024 --name dtwsurrogate_cnn_200epc --dtw --model cnn --dtw --best-loss
'''
# Get Dataloader, Model
name = args.name
train_loader, val_loader, test_loader = get_dtwdata(args)
test_iterator = get_data_iterator(test_loader)
device = set_devices(args)
sdtw = SoftDTW(use_cuda=True, gamma=0.1)

model = get_model(args, device=device)
evaluator = Evaluator(args)
criterion = get_loss(args)
DTW_MATRIX_TEST = np.load('./stores/dtw_matrix_test.npy')

# Check if result exists
result_ckpt = os.path.join(args.dir_result, name, 'test_result.pth')
if (not args.reset) and os.path.exists(result_ckpt):
    print('this experiment has tested before.')
    sys.exit()

# Check if checkpoint exists
if args.last:
    ckpt_path = os.path.join(args.dir_result, name, 'ckpts/last.pth')
if args.best_loss:
    ckpt_path = os.path.join(args.dir_result, name, 'ckpts/bestloss.pth')

if not os.path.exists(ckpt_path):
    print("invalid checkpoint path : {}".format(ckpt_path))

# Load checkpoint, model
ckpt = torch.load(ckpt_path, map_location=device)
state = ckpt['model']
model.load_state_dict(state)
model.eval()

evaluator.reset()
if args.plot_prob:
    prob = []
    label = []

with torch.no_grad():
    for global_step in tqdm(range(1, args.epochs*len(train_loader)//2 + 1), total=args.epochs*len(train_loader)//2 + +1, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
        dtw_test = []
        ecg1, ecg1_idx = next(test_iterator)
        ecg2, ecg2_idx = next(test_iterator)

        ecg1 = ecg1.to(device)
        ecg2 = ecg2.to(device)
        
        for i in range(args.batch_size):
            idx1 = ecg1_idx[i]
            idx2 = ecg2_idx[i]
            
            if DTW_MATRIX_TEST[idx1][idx2] == 0:
                dtw = calc_dtw_lead(sdtw, ecg1[i, :, :], ecg2[i, :, :])
                DTW_MATRIX_TEST[idx1][idx2] = dtw
                DTW_MATRIX_TEST[idx2][idx1] = dtw
            else:
                dtw = DTW_MATRIX_TEST[idx1][idx2]
            dtw_test.append(dtw)

        logits = model(ecg1, ecg2)
        loss = criterion(logits.squeeze(1).float(), torch.FloatTensor(dtw_test).to(device))
        evaluator.add_batch(torch.FloatTensor(dtw_test), logits.cpu(), loss, test=True)
        loss, r, pval = evaluator.performance_metric()
    print ('loss: {}, pearsonr: {}, pval : {}'.format(loss, r, pval))
    result_dict = {'rmse': loss}

np.save('stores/dtw_matrix_test.npy', DTW_MATRIX_TEST)
torch.save(result_dict, result_ckpt)
