#%%
%matplotlib inline
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 16
import numpy as np
import os
import pandas as pd
import seaborn as sns
from keras.applications import xception
from keras.preprocessing import image
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

#%%
# 定義
CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
NUM_CATEGORIES = len(CATEGORIES)
SEED = 1987
# pathの設定
data_dir = '/Users/zakopuro/Documents/kaggle/Plant_Seedlings/input/plant-seedlings-classification/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
sample_submission = pd.read_csv(os.path.join(data_dir,'sample_submission.csv'))
sample_submission.head(5)

#%%
for category in CATEGORIES:
	print('{} {} images'.format(category,len(os.listdir(os.path.join(train_dir,category)))))

#%%
train = []
for category_id , category in enumerate(CATEGORIES):
	for file in os.listdir(os.path.join(train_dir,category)):
		train.append(['train/{}/{}'.format(category,file),category_id,category])
train = pd.DataFrame(train,columns=['file','category_id','category'])
train.shape

#%%
# カテゴリー毎に200個ずつ取ってくる
SAMPLE_PER_CATEGORY = 200
train = pd.concat([train[train['category'] == c][:SAMPLE_PER_CATEGORY] for c in CATEGORIES])
train = train.sample(frac=1)
train.index = np.arange(len(train))
train.shape

#%%
test = []
for file in os.listdir(test_dir):
	test.append(['test/{}'.format(file),file])
test = pd.DataFrame(test,columns=['filepath','file'])
test.head(5)

#%%
def read_img(filepath,size):
	img = image.load_img(os.path.join(data_dir,filepath),target_size=size)
	img = image.img_to_array(img)
	return img

#%%
# fig = plt.figure(1,figsize=(NUM_CATEGORIES,NUM_CATEGORIES))
fig = plt.figure(1,figsize=(5,5))
grid = ImageGrid(fig,111,nrows_ncols=(NUM_CATEGORIES,NUM_CATEGORIES),axes_pad=0.05)
# i = 0
# for category_id, category in enumerate(CATEGORIES):
#     for filepath in train[train['category'] == category]['file'].values[:NUM_CATEGORIES]:
#         ax = grid[i]
#         img = read_img(filepath, (224, 224))
#         ax.imshow(img / 255.)
#         ax.axis('off')
#         if i % NUM_CATEGORIES == NUM_CATEGORIES - 1:
#             ax.text(250, 112, filepath.split('/')[1], verticalalignment='center')
#         i += 1
# plt.show();

#%%