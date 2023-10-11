import os
import pandas as pd
from torchvision.io import read_image

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
import cv2
ROOT_PATH='kaggle_room_street_data/street_data'
LABEL_DIR='images'
INPUT_DIR='images_BW'

#データフレーム作成
train_df=pd.DataFrame()
#カスタムデータセットの作成
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,img_bw_dir,img_label_dir,transform=None,target_transform=None):

        self.img_bw_dir=img_bw_dir#白黒画像の保存先
        self.img_label_dir=img_label_dir#ラベル画像(正解画像でありカラー元の画像)の保存先
        
        self.transform=transform
        self.target_transform=target_transform
    
    def __len__(self):
        sample=os.listdir(self.img_label_dir)
        return(len(sample))

    def __getitem__(self,idx):
        bw_list=os.listdir(self.img_bw_dir)
        imgBW_path=os.path.join(self.img_bw_dir,bw_list[idx])
        image=read_image(imgBW_path)
        plt.imshow(image.permute(1,2,0))
        img=cv2.imread(bw_list[idx])
        if self.idx==1:
            print(bw_list[idx])
        if self.transform:
            image=self.transform()

        label_list=os.listdir(self.img_label_dir)
        label_path=os.path.join(self.img_label_dir,label_list[idx])
        label=read_image(label_path)
        
        if self.target_transform:
            label=self.target_transform()
        sample={"image":image,"label":label}
        return sample

train_data=ImageDataset(
    img_bw_dir=f'{INPUT_DIR}/train/all',
    img_label_dir=f'{LABEL_DIR}/train/all',
    transform=ToTensor,#これにパラメータを[0,1]の間の値にする操作が含まれる
    target_transform=ToTensor


)
test_data=ImageDataset(
    img_bw_dir=f'{INPUT_DIR}/test/all',
    img_label_dir=f'{LABEL_DIR}/test/all',
    transform=ToTensor,
    target_transform=ToTensor

)

BATCH_SIZE=32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
#train_dataloader,valid_dataloader=torch.utils.data.random_split(dataset=train_dataloader,lengths=[125,16],generator=torch.Generator().manual_seed(42))
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

import torch.nn as nn
#ここからmodelを作成
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        #Encoder
        self.layer1=nn.Sequential(
            nn.Conv2d(1 ,16 ,kernel_size=3 ,padding='same'),
            nn.Conv2d(16,16,kernel_size=3,padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2)
        )
        self.layer2=nn.Sequential(
            
            nn.Conv2d(16,32,kernel_size=3,padding='same'),
            nn.Conv2d(32,32,kernel_size=3,padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2)
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding="same"),
            nn.Conv2d(64,64,kernel_size=3,padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        #decoder
        self.layer4_1=nn.Sequential(
            nn.Upsample(scale_factor=2,mode="bilinear"),
            #nn.ConvTranspose2d(64,32,kernel_size=3,)
            
        )
        self.layer4_2=nn.Sequential(
            nn.Conv2d(64,32,kernel_size=3,padding="same"),
            nn.Conv2d(32,16,kernel_size=3,padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer5_1=nn.Sequential(
            nn.Upsample(scale_factor=2,mode="bilinear")
            #nn.ConvTranspose2d(64,32,kernel_size=3,)
        )
        self.layer5_2=nn.Sequential(
            nn.Conv2d(32,16,kernel_size=3,padding="same"),
            nn.Conv2d(16,3,kernel_size=3,padding="same"),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

    def forward(self,x):

        x=self.layer1(x)
        x1=x
        
        x=self.layer2(x)
        x2=x 

        x=self.layer3(x)
        x3=x
        

        x=self.layer4_1(x)
        x4=x
        x=torch.cat([x2,x4],dim=1)
        x=self.layer4_2(x)

        x=self.layer5_1(x) 
        x5=x     
        x=torch.cat([x1,x5])
        x=self.layer5_2(x)

        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = Unet().to(device)
optimizer = optim.Adam(unet.parameters(), lr=0.001)
criterion=nn.MSELoss()

#学習を行う
history = {"train_loss": []}
n = 0
m = 0
EPOCHS=10

#print(train_dataloader)
for epoch in range(EPOCHS):
    train_loss=0
    val_loss=0
    unet.train()
    for idx ,data in enumerate(train_dataloader):
        inputs,labels=data["image"].to(device),data["label"].to(device)
        optimizer.zero_grad()
        outputs=unet(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        history["train_loss"].append(loss.item())
        n+=1
        if i%((len(train_data)//BATCH_SIZE)//10)==(len(train_data)//BATCH_SIZE)//10-1:
            print(f"epoch:{epoch+1}  index:{i+1}  train_loss:{train_loss/n:.5f}")
            n = 0
            train_loss = 0
            train_acc = 0



