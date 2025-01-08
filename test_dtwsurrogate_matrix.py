import itertools
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
CUDA_VISIBLE_DEVICES=1 python test_dtwsurrogate_matrix.py --ecg-len 1024 --name dtwsurrogate_matrix_cnn_200epc --dtw --matrix --model cnn --num-classes 2016 --last
'''
# Get Dataloader, Model
name = args.name
train_loader, val_loader, test_loader = get_dtwdata(args)
device = set_devices(args)
sdtw = SoftDTW(use_cuda=True, gamma=0.1)

model = get_model(args, device=device, dtw=False)
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
    for ecg, ecg_idx in tqdm(test_loader, total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
        idxs = list(ecg_idx)
        dtw_test = []
        ecg = ecg.to(device)
       
        for idx1, idx2 in itertools.combinations(ecg_idx, 2):
            if DTW_MATRIX_TEST[idx1][idx2] == 0:
                id1 = idxs.index(idx1)
                id2 = idxs.index(idx2)

                dtw = calc_dtw_lead(sdtw, ecg[id1, :, :], ecg[id2, :, :])
                DTW_MATRIX_TEST[idx1][idx2] = dtw
                DTW_MATRIX_TEST[idx2][idx1] = dtw
                dtw_test.append(dtw)
            else:
                dtw_test.append(DTW_MATRIX_TEST[idx1][idx2])

        logits = model(ecg)
#         import ipdb; ipdb.set_trace()
        
        loss = criterion(logits.float(), torch.FloatTensor(dtw_test).to(device))
        evaluator.add_batch(torch.FloatTensor(dtw_test), logits.cpu(), loss, test=True)
    loss, r, pval = evaluator.performance_metric()
    print ('loss: {}, pearsonr: {}, pval : {}'.format(loss, r, pval))
    result_dict = {'rmse': loss}

np.save('stores/dtw_matrix_test.npy', DTW_MATRIX_TEST)
torch.save(result_dict, result_ckpt)
