import os
import numpy as np
from tqdm import tqdm
from dtaidistance import dtw_ndim

import torch
from torch.utils.data import DataLoader

from data.unl_ecg import UNLECGDataset
from config import args
from model import get_model
from utils.utils import set_devices

'''
Calculating with DTW Surrogate Model
CUDA_VISIBLE_DEVICES=3 python save_dtw_batch.py --batch-size 64 --dtw --data-idx 8
python save_dtw_batch.py --batch-size 64 --data-idx 1
python save_dtw_batch.py --dist-calc euclidean --batch-size 64 --data-idx 1

Saving triplets for each 
'''
if args.dtw:
    save_dir = args.dir_dtw
elif args.dist_calc == 'euclidean':
    save_dir = args.dir_euc
else:
    save_dir = args.dir_dtw_real
    
device = set_devices(args)
# ecg_ids = np.load(os.path.join(args.dir_unl, "ecgno_batch.npy"))[(args.data_idx-1)*5000*args.batch_size: args.data_idx*5000*args.batch_size]
ecg_ids = np.load(os.path.join(args.dir_unl, "ecgno_batch.npy"))[5000*args.batch_size*16:]

data_loader = DataLoader(
    UNLECGDataset(args, ecg_ids),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=2,
)

if args.dtw: # calculate dtw with surrogate model
    dtw_ckpt = torch.load(os.path.join(args.dir_unl, args.surr_path), map_location=device)
    state = dtw_ckpt['model']
    dtw_model = get_model(args, device=device, dtw=True)
    dtw_model.load_state_dict(state)
    dtw_model.eval()

def calculate_dtw(train_x):
    batch_len = len(train_x)
    dist_matrix = torch.zeros((batch_len, batch_len))

    for i in range(batch_len):
        for j in range(i, batch_len):
            if i == j: 
                continue
            if dist_matrix[i][j] == 0:
                if args.dtw:
                    dtw_val = dtw_model(train_x[i].unsqueeze(0), train_x[j].unsqueeze(0)).cpu().detach()
                    dist_matrix[i][j] = dtw_val
                    dist_matrix[j][i] = dtw_val
                else:
                    ecg_i = train_x[i].numpy()
                    ecg_j = train_x[j].numpy()
                    dtw = dtw_ndim.distance(ecg_i, ecg_j)
                    dist_matrix[i][j] = dtw
                    dist_matrix[j][i] = dtw

    return dist_matrix

def calculate_euclidean(train_x):
    batch_len = len(train_x)
    dist_matrix = torch.zeros((batch_len, batch_len))
    
    for i in range(batch_len):
        for j in range(i, batch_len):
            if dist_matrix[i][j] == 0:
                dist = torch.mean(torch.sqrt((train_x[i]-train_x[j])**2))
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
    return dist_matrix

for step, train_batch in enumerate(tqdm(data_loader)):
    if args.dtw:
        train_x, idx = train_batch
        train_x = train_x.to(device)
    else:
        train_x = train_batch

    if args.dist_calc == 'euclidean':
        # torch.save(calculate_euclidean(train_x), os.path.join(save_dir, str((args.data_idx-1)*5000+step)+'.pth'))
        torch.save(calculate_euclidean(train_x), os.path.join(save_dir, str(80000+step)+'.pth'))
    else:
        # torch.save(calculate_dtw(train_x), os.path.join(save_dir, str((args.data_idx-1)*5000+step)+'.pth'))
        torch.save(calculate_dtw(train_x), os.path.join(save_dir, str(80000+step)+'.pth'))
        # torch.save(calculate_dtw(train_x), os.path.join(save_dir, str(84789)+'.pth'))
    
print("\n Finished calculating DTW with model")