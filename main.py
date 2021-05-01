### Script parameters
EXP_FOLDER = './'
USE_GPU = True

## Data set parameters
DATA_SET = 'COST2100'  # data set name
# 'COST2100'
# 'KWAUG14' or 'KWAUG14flat'
# 'SPACE08' or 'SPACE08flat'

## Network parameters
Z_DIM = 128  # dim of abstract representation vector

## Training parameters
TRAIN_MODEL = True

#########################################################################
### Initiate environment
if USE_GPU:
    pass
else:
    device = 'cpu'

### Load data
if DATA_SET == 'COST2100':
    pass
elif DATA_SET == 'KWAUG14':
    pass
elif DATA_SET == 'KWAUG14flat':
    pass
elif DATA_SET == 'SPACE08':
    pass
elif DATA_SET == 'SPACE08flat':
    pass
else:
    print("Data set not found")




### Result processing



