import torch
from torch.utils.data import DataLoader


import time
import os
import scipy.io as sio
import numpy as np


### Script parameters
EXP_FOLDER = './'
USE_GPU = True
MAN_SEED = None    # cuda manual seed
VERBOSE = True


## Model parameters
MODEL_TYPE = 'CRNet'
# 'CRNet' or 'CsiNet'

## Data set parameters
DATA_SET = 'COST2100'   # data set name
# 'COST2100'
# 'KWAUG14' or 'KWAUG14flat'
# 'SPACE08' or 'SPACE08flat'
# 'MNIST'
ENV_TYPE = 'indoor'   # for COST2100 only
# 'indoor' or 'outdoor'
BATCH_SIZE = 200
SAVE_INPUT = True


## Network parameters
Z_DIM = 128  # dim of abstract representation vector

## Training parameters
TRAIN_MODEL = True

#########################################################################
### Initiate environment
## Initiate result folders
data_path = './DataFiles/' + DATA_SET
res_path = EXP_FOLDER + 'ResultFiles/' + time.strftime('%y_%m_%d_%H%M%S_') + MODEL_TYPE + '_dim' + str(Z_DIM)
if not os.path.exists(res_path):
    os.makedirs(res_path)
if VERBOSE:
    if os.path.exists(data_path):
        print('\nLoad data from:' + data_path)
    else:
        print('\nData files not found')
    print('Results saved in: ' + res_path)

## Initiate device
if USE_GPU:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Running on GPU')
    else:
        print('GPU not found')
        device = torch.device('cpu')
        print('Running on CPU')
else:
    device = torch.device('cpu')
    print('Running on CPU')

### Load data
tmp_time = time.time()
if VERBOSE:
    print('\nStart loading data')
if DATA_SET == 'COST2100':
    img_channels = 2
    img_height = 32
    img_width = 32
    if ENV_TYPE == 'indoor':
        mat = sio.loadmat(data_path + '/DATA_Htrainin.mat')
        x_train = mat['HT']  # array
        mat = sio.loadmat(data_path + '/DATA_Hvalin.mat')
        x_val = mat['HT']  # array
        mat = sio.loadmat(data_path + '/DATA_Htestin.mat')
        x_test = mat['HT']  # array
        mat = sio.loadmat(data_path + '/DATA_HtestFin_all.mat')
        X_test = mat['HF_all']  # array
    elif ENV_TYPE == 'outdoor':
        mat = sio.loadmat(data_path + '/DATA_Htrainout.mat')
        x_train = mat['HT']  # array
        mat = sio.loadmat(data_path + '/DATA_Hvalout.mat')
        x_val = mat['HT']  # array
        mat = sio.loadmat(data_path + '/DATA_Htestout.mat')
        x_test = mat['HT']  # array
        mat = sio.loadmat(data_path + '/DATA_HtestFout_all.mat')
        X_test = mat['HF_all']  # array

    x_train = torch.tensor(x_train, dtype=torch.float32).view(len(x_train), img_channels, img_height, img_width)
    x_val = torch.tensor(x_val, dtype=torch.float32).view(len(x_val), img_channels, img_height, img_width)
    x_test = torch.tensor(x_test, dtype=torch.float32).view(len(x_test), img_channels, img_height, img_width)
    print(np.shape(X_test))


    if SAVE_INPUT:
        x_train_np = x_train.detach().numpy()
        x_train_real = x_train_np[:, 0, :, :] - 0.5
        x_train_real = x_train_real.reshape((len(x_train_np), img_height, img_width))
        x_train_imag = x_train_np[:, 1, :, :] - 0.5
        x_train_imag = x_train_imag.reshape((len(x_train_np), img_height, img_width))
        x_train_np = x_train_real + 1j * x_train_imag
        sio.savemat(res_path + '/x_train_cplx.mat', {'x_train': x_train_np})
        x_val_np = x_val.detach().numpy()
        x_val_real = x_val_np[:, 0, :, :] - 0.5
        x_val_real = x_val_real.reshape((len(x_val_np), img_height, img_width))
        x_val_imag = x_val_np[:, 1, :, :] - 0.5
        x_val_imag = x_val_imag.reshape((len(x_val_np), img_height, img_width))
        x_val_np = x_val_real + 1j * x_val_imag
        sio.savemat(res_path + '/x_val_cplx.mat', {'x_val': x_val_np})
        x_test_np = x_test.detach().numpy()
        x_test_real = x_test_np[:, 0, :, :] - 0.5
        x_test_real = x_test_real.reshape((len(x_test_np), img_height, img_width))
        x_test_imag = x_test_np[:, 1, :, :] - 0.5
        x_test_imag = x_test_imag.reshape((len(x_test_np), img_height, img_width))
        x_test_np = x_test_real + 1j * x_test_imag
        sio.savemat(res_path + '/x_test_cplx.mat', {'x_test': x_test_np})
        sio.savemat(res_path + '/X_test_cplx.mat', {'X_test': X_test_np})
        if VERBOSE:
            print('Input data saved as .mat files')

elif DATA_SET == 'KWAUG14':
    print(DATA_SET + ': Not ready yet')
    pass
elif DATA_SET == 'KWAUG14flat':
    print(DATA_SET + ': Not ready yet')
    pass
elif DATA_SET == 'SPACE08':
    print(DATA_SET + ': Not ready yet')
    pass
elif DATA_SET == 'SPACE08flat':
    print(DATA_SET + ': Not ready yet')
    pass
else:
    print("Data set not found")

train_loader = DataLoader



if VERBOSE:
    print('Data loaded in:' + str(time.time() - tmp_time) + ' s')


### Result processing


