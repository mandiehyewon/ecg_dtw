'''
Modified: 
Code for calculating DTW value before feeding into the model
Calculating the actual DTW value with softDTW module
CUDA_VISIBLE_DEVICES=1 python dtw_calculate.py --dtw --model cnn
'''

import torch
import os
import h5py
import hdf5plugin
import time
import enlighten
from typing import List
from tqdm import tqdm

import numpy as np
from model import get_model
from config import args


ALL_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
def load_ecg(
    hd5: h5py.File, date: str,
    target_length: int = 2500,
    leads: List[str] = ALL_LEADS,
):
    out = np.empty((target_length, len(leads)))
    for i, lead in enumerate(leads):
        lead_array = hd5["ecg"][date][lead][()]
        out[:, i] = np.interp(
            np.linspace(0, 1, target_length),
            np.linspace(0, 1, lead_array.shape[0]),
            lead_array,
        )
    return out

def calc_dtw_lead(dtw_model, ecg1, ecg2):
    dtw_d = []
    for lead in range(12):
        x = ecg1[:, lead].unsqueeze(0).unsqueeze(2)
        y = ecg2[:, lead].unsqueeze(0).unsqueeze(2)
        
        d = dtw_model(x, y)
        dtw_d.append(d.cpu().detach().numpy())
    
    return np.mean(dtw_d)

if __name__ == "__main__":
    # Load ECG IDs
    ecgno = np.load('./stores/ecgno.npy')
    pt_no = len(ecgno)
    ecg_len = 2500

    # Load model
    device = torch.device('cuda')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    dtw_ckpt = torch.load(os.path.join(args.dir_result, args.surr_path), map_location=device)
    state = dtw_ckpt['model']
    dtw_model = get_model(args, device=device, dtw=True)
    dtw_model.load_state_dict(state)
    dtw_model.eval()
    print('loaded model')
    import ipdb; ipdb.set_trace()

    # calculate dtw using softdtw model
    f = h5py.File('dtw_matrix.h5', 'w')
    dtw_data = f.create_dataset("surrogate", (pt_no, pt_no), maxshape=(None, None))
    print('loaded hd5')

    # Load enlighten progress bar
#     manager = enlighten.get_manager()
#     cols = manager.counter(total=pt_no, desc="Cols", unit="ecgs", color="red")
#     rows = manager.counter(total=pt_no, desc="Rows", unit="ecgs", color="blue")
#     print('loaded enlighten manager')

    # Loading ECGs        
    for i in tqdm(range(pt_no)):
        ecg1_id = ecgno[i].split('_')
        ecg1_hd5 = h5py.File(os.path.join(args.dir_unl, ecg1_id[0], ecg1_id[1]+'.hd5'), 'r')
        ecg1 = (load_ecg(ecg1_hd5, ecg1_id[2], ecg_len).astype(np.float32) / 1000).T
        ecg1 = torch.from_numpy(ecg1).to(device)
        
#         cols.update()
        for j in range(i):
            if f['surrogate'][i][j] == 0:
                ecg2_id = ecgno[j].split('_')
                ecg2_hd5 = h5py.File(os.path.join(args.dir_unl, ecg2_id[0], ecg2_id[1]+'.hd5'), 'r')
                ecg2 = (load_ecg(ecg2_hd5, ecg2_id[2], ecg_len).astype(np.float32) / 1000).T #.to(device)
                ecg2 = torch.from_numpy(ecg2).to(device)
                
                dtw_val = calc_dtw_lead(dtw_model, ecg1, ecg2)
                f['surrogate'][i][j] = dtw_val
                f['surrogate'][j][i] = dtw_val
#             rows.update()

#     manager.stop()
    print('finished calculating')
    f.close()

    num_processes = 8
    dtw_model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=calculate, args=(dtw_model,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        
# with h5py.File('test.hdf', 'w') as outfile:
#     dset = outfile.create_dataset('a_descriptive_name', data=data, chunks=True)
#     dset.attrs['some key'] = 'Did you want some metadata?'