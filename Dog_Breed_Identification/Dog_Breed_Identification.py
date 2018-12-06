#%%
%matplotlib inline
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.applications import xception
from keras.preprocessing import image
from tqdm import tqdm


#%%
data_dir = '/Users/zakopuro/Code/python_code/kaggle/Dog_Breed_Identification/input/'
train_dir = os.path.join(data_dir,'train')
test_dri  = os.path.join(data_dir,'test')
sample_submission = pd.read_csv(os.path.join(data_dir,'sample_submission.csv'))
labels = pd.read_csv(os.path.join(data_dir,'labels.csv'))
labels.head(5)

#%%
breed = labels['breed']
BREED = breed[~breed.duplicated()]
NUM_BREED = len(BREED)