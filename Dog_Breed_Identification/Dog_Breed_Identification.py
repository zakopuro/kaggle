#%%
%matplotlib inline
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from keras.applications import xception
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
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
# ResNet50を利用してエラーチェック
# 画像で確認
model = ResNet50(weights='imagenet')
fig = plt.figure(1,figsize =(30,30))
grid = ImageGrid(fig,111,nrows_ncols=(10,5),axes_pad=0.05)
i = 0
for breed_id,breed in enumerate(BREED[0:10]):
	for filepath in train[train['breed'] == breed]['file_path'].values[:5]:
		ax = grid[i]
		img = read_img(filepath,(224,224))
		ax.imshow(img / 255.)
		x = preprocess_input(np.expand_dims(img.copy(), axis=0))
		preds = model.predict(x)
		_, imagenet_class_name, prob = decode_predictions(preds, top=1)[0][0]
		ax.text(10, 180, 'ResNet50: %s (%.2f)' % (imagenet_class_name , prob), color='w', backgroundcolor='k', alpha=0.8)
		ax.text(10, 210, 'LABEL: %s' % breed, color='k', backgroundcolor='w', alpha=0.8)
		ax.axis('off')
		if i % 5 == 5 - 1:
			ax.text(250,112, breed , verticalalignment ='center')
		i += 1
plt.show();

# %%
model = ResNet50(weights='imagenet')
ReNet_ans_num = 0
for breed_id,breed in enumerate(BREED):
	for filepath in train[train['breed'] == breed]['file_path'].values:
		img = read_img(filepath,(224,224))
		ax.imshow(img / 255.)
		x = preprocess_input(np.expand_dims(img.copy(), axis=0))
		preds = model.predict(x)
		_, imagenet_class_name, prob = decode_predictions(preds, top=1)[0][0]
		if imagenet_class_name.upper() == breed.upper():
			ReNet_ans_num +=  1
acc = (ReNet_ans_num * 100) /len(train)
print('正答数:',ReNet_ans_num,'\n','正答率:',acc)
