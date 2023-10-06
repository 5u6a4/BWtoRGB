import os
from sklearn.model_selection import train_test_split
import shutil
import torch
from torch.utils.data import DataLoader
#画像が保存されているディレクトリのパス
ROOT_PATH='kaggle_room_street_data/street_data'
#root_path='C:/Users/mutsu/venv/kaggle_room_street_data/street_data'
INPUT_DIR='images'
TRAIN_PATH=f'{INPUT_DIR}/train'
TEST_PATH=f'{INPUT_DIR}/test'

street_list=list()
if not os.path.exists(INPUT_DIR):
    os.mkdir(TRAIN_PATH)
    os.mkdir(TEST_PATH)

num_images_to_use=5000
names=os.listdir(ROOT_PATH)[:num_images_to_use]
train_names,test_names=train_test_split(names,test_size=0.1,random_state=10)
print(len(train_names))
print(len(test_names))

#ファイルのコピー
# image files send to TRAIN_PATH 
for img in train_names:
    if 'apartment' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TRAIN_PATH}/apartment',img))
    if 'church' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TRAIN_PATH}/church',img))
    if 'garage' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TRAIN_PATH}/garage',img))
    if 'house' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TRAIN_PATH}/house',img))
    if 'industrial' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TRAIN_PATH}/industrial',img))
    if 'office' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TRAIN_PATH}/office',img))
    if 'retail' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TRAIN_PATH}/retail',img))
    if 'roof' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TRAIN_PATH}/roof',img))


for img in test_names:
    if 'apartment' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TEST_PATH}/apartment',img))
    if 'church' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TEST_PATH}/church',img))
    if 'garage' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TEST_PATH}/garage',img))
    if 'house' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TEST_PATH}/house',img))
    if 'industrial' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TEST_PATH}/industrial',img))
    if 'office' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TEST_PATH}/office',img))
    if 'retail' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TEST_PATH}/retail',img))
    if 'roof' in img:
        shutil.copyfile(os.path.join(ROOT_PATH,img),os.path.join(f'{TEST_PATH}/roof',img))

