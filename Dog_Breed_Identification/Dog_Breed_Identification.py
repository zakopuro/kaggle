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
from mpl_toolkits.axes_grid1 import ImageGrid

#%%
data_dir = '/Users/zakopuro/Code/python_code/kaggle/Dog_Breed_Identification/input/'
train_dir = os.path.join(data_dir,'train')
test_dir  = os.path.join(data_dir,'test')
sample_submission = pd.read_csv(os.path.join(data_dir,'sample_submission.csv'))
labels = pd.read_csv(os.path.join(data_dir,'labels.csv'))
labels.head(5)

#%%
labels.sort_values('breed',inplace = True)
breed = labels['breed']
BREED = breed[~breed.duplicated()]
NUM_BREED = len(BREED)
# それぞれの犬の画像の数を表示
breeds_num = []
for breed in BREED:
	breeds_num.append(len(labels[labels['breed'].isin([breed])]))
	print('{} {} images'.format(breed,len(labels[labels['breed'].isin([breed])])))
# 一番少ない画像の数
SAMPLE_PER_BREED = min(breeds_num)

#%%
train = labels.copy()
train = train.rename(columns={'id':'file_path'})
train
# ADD breed_id
for breed_id,breed in enumerate(BREED):
	train.loc[train['breed'] == breed, 'breed_id'] = breed_id
train.sort_values('breed_id',inplace = True)
train.reset_index(drop = True,inplace = True)
train['file_path'] = 'train/' + train['file_path'] + '.jpg'
train.head()


#%%
# 少ない画像の枚数に合わせる
train = pd.concat([train[train['breed'] == c][:SAMPLE_PER_BREED] for c in BREED])
train = train.sample(frac=1)
train.index = np.arange(len(train))
train.head()
train.shape


#%%
test = []
for file in os.listdir(test_dir):
    test.append(['test/{}'.format(file), file])
test = pd.DataFrame(test, columns=['filepath', 'file'])
test.head()
test.shape

#%%
def read_img(filepath, size):
    img = image.load_img(os.path.join(data_dir, filepath), target_size=size)
    img = image.img_to_array(img)
    return img

#%%
fig = plt.figure(1,figsize =(20,20))
grid = ImageGrid(fig,111,nrows_ncols=(10,5),axes_pad=0.05)
i = 0
for breed_id in range(10):
	for filepath in train[train['breed_id'] == breed_id]['file_path'].values[:5]:
		ax = grid[i]
		img = read_img(filepath,(224,224))
		ax.imshow(img / 255.)
		ax.axis('off')
		if i % 5 == 5 - 1:
			ax.text(250,112, train['breed'][breed_id] , verticalalignment ='center')
		i += 1
plt.show();

#%%
