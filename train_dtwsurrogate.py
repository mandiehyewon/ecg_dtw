import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from config import args
from data import get_dtwdata, get_data_iterator
from model import get_model
from utils.loss import get_loss
from utils.metrics import Evaluator
from utils.logger import Logger
from utils.utils import set_seeds, set_devices, calc_dtw_lead
from utils.lr_scheduler import LR_Scheduler
from utils.soft_dtw_cuda import SoftDTW

'''
CUDA_VISIBLE_DEVICES=3 python train_dtwsurrogate.py --ecg-len 1024 --train-mode regression --model cnn --dtw --epoch 200 --val-iter 1000 --name dtwsurrogate_cnn_200epc
'''

seed = set_seeds(args)
device = set_devices(args)
logger = Logger(args)
sdtw = SoftDTW(use_cuda=True, gamma=0.1)

# Load Data, Create Model
train_loader, val_loader, test_loader = get_dtwdata(args)
train_iterator = get_data_iterator(train_loader)
val_iterator = get_data_iterator(val_loader)
model = get_model(args, device=device)
DTW_MATRIX_TRAIN = np.load('./stores/dtw_matrix_train.npy')
DTW_MATRIX_VALID = np.load('./stores/dtw_matrix_valid.npy')

criterion = get_loss(args)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = LR_Scheduler(optimizer, args.scheduler, args.lr, args.epochs, from_iter=args.lr_sch_start, warmup_iters=args.warmup_iters, functional=True)

### TRAINING
pbar = tqdm(total=args.epochs*len(train_loader)//2 + 1, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
for global_step in range(1, args.epochs*len(train_loader)//2 + 1):
    dtw_batch = []
    loss = 0
    ecg1, ecg1_idx = next(train_iterator)
    ecg2, ecg2_idx = next(train_iterator)

    ecg1 = ecg1.to(device)
    ecg2 = ecg2.to(device)

    # Compute the dtw value
    for i in range(args.batch_size):
        idx1 = ecg1_idx[i]
        idx2 = ecg2_idx[i]
        
        if DTW_MATRIX_TRAIN[idx1][idx2] == 0:
            dtw = calc_dtw_lead(sdtw, ecg1[i, :, :], ecg2[i, :, :])
            DTW_MATRIX_TRAIN[idx1][idx2] = dtw
            DTW_MATRIX_TRAIN[idx2][idx1] = dtw
            dtw_batch.append(dtw)
        else:
            dtw_batch.append(DTW_MATRIX_TRAIN[idx1][idx2])

    logits = model(ecg1, ecg2)
    loss = criterion(logits.squeeze(1).float(), torch.FloatTensor(dtw_batch).to(device))
    logger.loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ## LOGGING
    if global_step % args.log_iter == 0:
        logger.log_tqdm(pbar)
        logger.log_scalars(global_step)
        logger.loss_reset()

    ### VALIDATION
    if global_step % args.val_iter == 0:
        dtw_val = []
        model.eval()
        logger.evaluator.reset()
        with torch.no_grad():
            ecgval_1, ecg1val_idx = next(val_iterator)
            ecgval_2, ecg2val_idx = next(val_iterator)

            ecgval1 = ecgval_1.to(device)
            ecgval2 = ecgval_2.to(device)

            for i in range(args.batch_size):
                idx1 = ecg1val_idx[i]
                idx2 = ecg2val_idx[i]

                if DTW_MATRIX_VALID[idx1][idx2] == 0:
                    dtw = calc_dtw_lead(sdtw, ecgval1[i, :, :], ecgval2[i, :, :])
                    DTW_MATRIX_VALID[idx1][idx2] = dtw
                    DTW_MATRIX_VALID[idx2][idx1] = dtw

                else:
                    dtw = DTW_MATRIX_VALID[idx1][idx2]
                dtw_val.append(dtw)
            logits = model(ecg1, ecg2)
            loss = criterion(logits.squeeze(1).float(), torch.FloatTensor(dtw_batch).to(device))
            logger.evaluator.add_batch(torch.FloatTensor(dtw_batch), logits.cpu(), loss)
            logger.add_validation_logs(global_step, loss)
        model.train()
    logger.save(model, optimizer, global_step)
    pbar.update(1)

ckpt = logger.save(model, optimizer, global_step, last=True)
logger.writer.close()

np.save('stores/dtw_matrix_train.npy', DTW_MATRIX_TRAIN)
np.save('stores/dtw_matrix_valid.npy', DTW_MATRIX_VALID)
print("\n Finished training.......... Please Start Testing with test.py")