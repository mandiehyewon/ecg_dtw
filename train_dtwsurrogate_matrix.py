import itertools
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
CUDA_VISIBLE_DEVICES=3 python train_dtwsurrogate.py --ecg-len 1024 --train-mode regression --model cnn --dtw --matrix --epoch 200 --val-iter 1000 --embedding-dim 2016 --name dtwsurrogate_matrix_cnn_200epc
'''

seed = set_seeds(args)
device = set_devices(args)
logger = Logger(args)
sdtw = SoftDTW(use_cuda=True, gamma=0.1)

# Load Data, Create Model
train_loader, val_loader, test_loader = get_dtwdata(args)
DTW_MATRIX_TRAIN = np.load('./stores/dtw_matrix_train.npy')
DTW_MATRIX_VALID = np.load('./stores/dtw_matrix_valid.npy')

model = get_model(args, device=device, dtw=False)
criterion = get_loss(args)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = LR_Scheduler(optimizer, args.scheduler, args.lr, args.epochs, from_iter=args.lr_sch_start, warmup_iters=args.warmup_iters, functional=True)

### TRAINING
pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
for epoch in range(1, args.epochs + 1):
    loss = 0
    for ecg, ecg_idx in train_loader:
        idxs = list(ecg_idx)
        dtw_batch = [] # len = 2016
        ecg = ecg.to(device)

        # Compute the dtw value
        for idx1, idx2 in itertools.combinations(ecg_idx, 2):
            if DTW_MATRIX_TRAIN[idx1][idx2] == 0:
                id1 = idxs.index(idx1)
                id2 = idxs.index(idx2)

                dtw = calc_dtw_lead(sdtw, ecg[id1, :, :], ecg[id2, :, :])
                DTW_MATRIX_TRAIN[idx1][idx2] = dtw
                DTW_MATRIX_TRAIN[idx2][idx1] = dtw
                dtw_batch.append(dtw)
            else:
                dtw_batch.append(DTW_MATRIX_TRAIN[idx1][idx2])

        logits = model(ecg)
        loss = criterion(logits.float(), torch.FloatTensor(dtw_batch).to(device))
        logger.loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ## LOGGING
    if epoch % args.log_iter == 0:
        logger.log_tqdm(pbar)
        logger.log_scalars(epoch)
        logger.loss_reset()

    ### VALIDATION
    if epoch % args.val_iter == 0:
        model.eval()
        logger.evaluator.reset()

        with torch.no_grad():
            for ecgval, ecgval_idx in val_loader:
                idxs = list(ecgval_idx)
                dtw_val = []
                ecgval = ecgval.to(device)
                
                for idx1, idx2 in itertools.combinations(ecgval_idx, 2):
                    if DTW_MATRIX_VALID[idx1][idx2] == 0:
                        id1 = idxs.index(idx1)
                        id2 = idxs.index(idx2)

                        dtw = calc_dtw_lead(sdtw, ecgval[id1, :, :], ecgval[id2, :, :])
                        DTW_MATRIX_VALID[idx1][idx2] = dtw
                        DTW_MATRIX_VALID[idx2][idx1] = dtw

                    else:
                        dtw = DTW_MATRIX_VALID[idx1][idx2]
                        dtw_val.append(dtw)

                logits = model(ecgval)
                loss = criterion(logits.float(), torch.FloatTensor(dtw_batch).to(device))
                logger.evaluator.add_batch(torch.FloatTensor(dtw_batch), logits.cpu(), loss)
                logger.add_validation_logs(epoch, loss)
        model.train()
    logger.save(model, optimizer, epoch)
    pbar.update(1)

ckpt = logger.save(model, optimizer, epoch, last=True)
logger.writer.close()

np.save('stores/dtw_matrix_train.npy', DTW_MATRIX_TRAIN)
np.save('stores/dtw_matrix_valid.npy', DTW_MATRIX_VALID)
print("\n Finished training.......... Please Start Testing with test.py")