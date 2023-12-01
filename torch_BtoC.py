import os
from pickle import TRUE
from random import shuffle
from PIL import Image
from torchvision.io import read_image

import torch
from torch.utils.data import Dataset
from torchvision import datasets

from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
ROOT_PATH='kaggle_room_street_data/street_data'
LABEL_DIR='images'
INPUT_DIR='images_BW'

import numpy as np


#カスタムデータセットの作成
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,img_bw_dir,img_label_dir,transform=None,target_transform=None):

        self.img_bw_dir=img_bw_dir#白黒画像の保存先
        self.img_label_dir=img_label_dir#ラベル画像(正解画像でありカラー元の画像)の保存先

        self.bw_list=os.listdir(self.img_bw_dir)
        self.label_list=os.listdir(self.img_label_dir)
        
        self.transform=transform
        self.target_transform=target_transform
    
    def __len__(self):
        # sample=os.listdir(self.img_label_dir)
        # return(len(sample))
        return len(self.label_list)

    def __getitem__(self,idx):
        imgBW_path=os.path.join(self.img_bw_dir,self.bw_list[idx])
        image=Image.open(imgBW_path)
        if self.transform:
            image=self.transform(image)

        
        label_path=os.path.join(self.img_label_dir,self.label_list[idx])
        label=Image.open(label_path)
        
        if self.target_transform:
            label=self.target_transform(label)
        sample={"image":image,"label":label}
        return sample

train_data=ImageDataset(
    img_bw_dir=f'{INPUT_DIR}/train/all',
    img_label_dir=f'{LABEL_DIR}/train/all',
    transform=ToTensor(),#これにパラメータを[0,1]の間の値にする操作が含まれる
    target_transform=ToTensor()
)
test_data=ImageDataset(
    img_bw_dir=f'{INPUT_DIR}/test/all',
    img_label_dir=f'{LABEL_DIR}/test/all',
    transform=ToTensor(),
    target_transform=ToTensor()
)


print(type(train_data))
BATCH_SIZE=32
TRAIN_SIZE=int(len(train_data)*0.9)
VALID_SIZE=int(len(train_data)-len(train_data)*0.9)
train,val=torch.utils.data.random_split(dataset=train_data,
lengths=[TRAIN_SIZE,VALID_SIZE])
train_dataloader=DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader=DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)

test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
print(type(test_dataloader))
import torch.nn as nn
#ここからmodelを作成
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        #Encoder
        self.layer1=nn.Sequential(
            nn.Conv2d(1 ,16 ,kernel_size=3 ,padding='same'),
            nn.Conv2d(16,16,kernel_size=3,padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2=nn.Sequential(
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(16,32,kernel_size=3,padding='same'),
            nn.Conv2d(32,32,kernel_size=3,padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer3=nn.Sequential(
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(32,64,kernel_size=3,padding="same"),
            nn.Conv2d(64,64,kernel_size=3,padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer4=nn.Sequential(
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(64,128,kernel_size=3,padding="same"),
            nn.Conv2d(128,128,kernel_size=3,padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        #decoder
        self.layer5_1=nn.Sequential(
            nn.Upsample(scale_factor=2,mode="bilinear"),
            #nn.ConvTranspose2d(128,64,kernel_size=3,stride=2),
            nn.Conv2d(128,64,kernel_size=3,padding="same")
        )
        self.layer5_2=nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer6_1=nn.Sequential(
            nn.Upsample(scale_factor=2,mode="bilinear"),
            #nn.ConvTranspose2d(64,32,kernel_size=3),
            nn.Conv2d(64,32,kernel_size=3,padding="same")
        )
        self.layer6_2=nn.Sequential(
            nn.Conv2d(64,32,kernel_size=3,padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer7_1=nn.Sequential(
            nn.Upsample(scale_factor=2,mode="bilinear"),
            #nn.ConvTranspose2d(32,16,kernel_size=3),
            nn.Conv2d(32,16,kernel_size=3,padding="same")
        )
        self.layer7_2=nn.Sequential(
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
        
        x=self.layer4(x)
        
        x=self.layer5_1(x)  
        x=torch.cat([x3,x],dim=1)
        x=self.layer5_2(x)
        x=self.layer6_1(x)  

        x=torch.cat([x2,x],dim=1)
        x=self.layer6_2(x)
        
        x=self.layer7_1(x)
        x=torch.cat([x1,x],dim=1)
        x=self.layer7_2(x)
        
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = UNET().to(device)
optimizer = optim.Adam(unet.parameters(), lr=0.0001)
criterion=nn.MSELoss()

#学習を行う
history_t = {"train_loss": []}
history_v={"valid_loss":[]}
n = 0
m = 0
EPOCHS=60


print(train_dataloader)
for epoch in range(EPOCHS):
    train_loss=0
    val_loss=0
    unet.train()
    first_t=True
    first_v=True
    for idx,sample in enumerate(train_dataloader):
        inputs,labels=sample["image"].to(device),sample["label"].to(device)
        optimizer.zero_grad()
        outputs=unet(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        
        n+=1
        if idx%((TRAIN_SIZE//BATCH_SIZE)//10)==(TRAIN_SIZE//BATCH_SIZE)//10-1:
            print(f"epoch:{epoch+1}  index:{idx+1}  train_loss:{train_loss/n:.5f}")
            if first_t is True:
                history_t["train_loss"].append(loss.item())
                first_t=False
            n = 0
            train_loss = 0
            train_acc = 0
    
    unet.eval()
    with torch.no_grad():
        for idx,sample in enumerate(valid_dataloader):
            inputs,labels=sample["image"].to(device),sample["label"].to(device)
            outputs=unet(inputs)
            loss = criterion(outputs,labels)
            val_loss+=loss.item()
            
            m+=1
            if idx%((VALID_SIZE//BATCH_SIZE)//10)==(VALID_SIZE//BATCH_SIZE)//10-1:
                print(f"epoch:{epoch+1}  index:{idx+1}  val_loss:{val_loss/m:.5f}")
                if first_v is True:
                    history_v["valid_loss"].append(loss.item())
                    first_v=False
                n = 0
                val_loss = 0
                train_acc = 0
    #torch.save(unet.state_dict(), f"./train_{epoch+1}.pth")
print("finish training")

plt.plot(np.arange(1, epoch+2),history_t["train_loss"],label="train")
plt.plot(np.arange(1, epoch+2),history_v["valid_loss"],label="valid")

plt.xlabel('EPOCH')
plt.ylabel('loss')
plt.show()

model=UNET()
model.load_state_dict(torch.load("./train_60.pth"))
model.eval()
with torch.no_grad():
    data=next(iter(test_dataloader))
    inputs,labels=data["image"],data["label"]
    outputs=model(inputs)
    loss=criterion(outputs,labels)
    print("loss:",loss.item())

ReLU=nn.ReLU()
outputs=ReLU(outputs)

classes = ["grayscale","to_color","original"]
PREDICTS=3
START=0

for j in range(10):
    
    fig, ax = plt.subplots(PREDICTS, len(classes), figsize=(15,15))
    for i in range(PREDICTS):
        ax[i,0].imshow(outputs[START+i,:,:,:].permute(1,2,0))
        # ax[i,0].set_title(f"pred")
        ax[i,0].axis("off")

        plt.gray()
        ax[i,1].imshow(data["image"][START+i,:,:,:].permute(1,2,0))
        # ax[i,1].set_title(f"gray")
        ax[i,1].axis("off")


        ax[i,2].imshow(data["label"][START+i,:,:,:].permute(1,2,0))
        # ax[i,2].set_title(f"original")
        ax[i,2].axis("off")
    START+=3
    plt.savefig(f'{j}'+'lr_10^-4_epoch60.png')
    plt.show()
    