from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import PIL
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Activation,BatchNormalization,Flatten,Dense
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
import shutil

import time
from IPython import display



#画像が保存されているディレクトリのパス
ROOT_PATH='kaggle_room_street_data/street_data'
#root_path='C:/Users/mutsu/venv/kaggle_room_street_data/street_data'
INPUT_DIR='images'
TRAIN_PATH=f'{INPUT_DIR}/train'
TEST_PATH=f'{INPUT_DIR}/test'

street_list=list()

INPUT_SIZE=(224,224)
BATCH_SIZE=32
EPOCHS=10

#訓練用画像データ
train_x=[]#BW
train_y=[]#trainの正解ラベル(カラー画像)
#検証用(チューニングするときに使う)
valid_x=[]
valid_y=[]
#modelの最終評価
test_x=[]
test_y=[]

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

print(TRAIN_PATH)
print(TEST_PATH)

#テンソル化とデータローダー作成
train_dataset=tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.2,
    subset='training',
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=(INPUT_SIZE[0],INPUT_SIZE[1]),
    seed=1,

)
#train_images=train_dataset.reshape(train_dataset.shape[0],224,224,3).astype('float32')
valid_dataset=tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.2,
    subset='validation',
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=(INPUT_SIZE[0],INPUT_SIZE[1]),
    seed=1
)
#valid_dataset=valid_dataset.reshape(valid_dataset.shape[0],28,28,1).astype('float32')
test=tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    shuffle=False,
    batch_size=BATCH_SIZE,

    image_size=(INPUT_SIZE[0],INPUT_SIZE[1]),
    )

print(type(train_dataset))
print(len(train_dataset))
for image_batch, labels_batch in train_dataset:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


# def CNN():
#     model=tf.keras.Sequential()
#     model.add(tf.layers.Rescaling(1./255)) #画像を正規化(0~255を0~1に))
#     model.add(tf.layers.Conv2D(16,(3,3),strides=(1,1),input_shape=[224,224,1])) #[224,224,1]->[222,222,16]
#     model.add(tf.layers.Conv2D(16,(3,3),strides=(1,1),input_shape=[222,222,16])) #[222,222,16]->[220,220,16]
#     model.add(tf.MaxPooling2D(pool_size=(2,2),strides=None))                   #[220,220,16]->[110,110,16]

#     model.add(tf.layers.Conv2D(32,(3,3),strides=(1,1),input_shape=[110,110,16])) #[110,110,16]->[108,108,32]
#     model.add(tf.layers.Conv2D(32,(3,3),strides=(1,1),input_shape=[110,110,16])) #[108,108,32]->[106,106,32]    
#     model.add(tf.MaxPooling2D(pool_size=(2,2),strides=None))                     #[106,106,32]->[53,53,64]

class Unet(Model):
    def __init__(self):
        super().__init__()
        self.enc=Encoder()
        self.dec=Decoder()
    
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        
        self.loss_object=tf.keras.losses.MeanSquaredError(name='mean_squared_error')
        self.train_loss=tf.keras.metrics.Mean('train_loss',dtype=tf.float32)
        self.valid_loss=tf.keras.metrics.Mean('valid_loss',dtype=tf.float32)
    
    def call(self,x):
        z1,z2,z3_dropout=self.enc(x)
        output_color=self.dec(z1,z2,z3_dropout)
        return output_color

    @tf.function 
    def train_step(self,input_color): #input_colorは入力画像のこと(224,224,3)これが正解ラベルになる
        with tf.GradientTape() as tape:

            output_color=self.call(input_color)
            criterion=self.loss_object(input_color,output_color)

        gradients=tape.gradient(criterion,self.trainable_variable)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
        self.train_loss(criterion)
    
    @tf.function
    def valid_step(self,input_color):
        y=self.call(input_color)
        v_loss=self.loss_object(input_color,y)
        self.valid_loss(v_loss)

        return y

class Encoder(Model):
    def __init__(self):
        super().__init__()
        inputs=tf.keras.Input(shape=(224,224,3))
        self.block1=tf.keras.layers.Rescaling(scale=1./255,name='block1')(inputs)
        self.block1_conv1=tf.keras.layers.Conv2D(16,(3,3),name='block1_conv1',strides=(1,1),padding='same',input_shape=[224,224,3])
        self.block1_conv2=tf.keras.layers.Conv2D(16,(3,3),name='block1_conv2',strides=(1,1),padding='same')
        self.block1_bn = tf.keras.layers.BatchNormalization()
        self.block1_act = tf.keras.layers.ReLU()
        self.block1_pool=tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,name='block1_pool')
        

        self.block2_conv1=tf.keras.layers.Conv2D(32,(3,3),name='block2_conv1',strides=(1,1),padding='same')
        self.block2_conv2=tf.keras.layers.Conv2D(32,(3,3),name='block2_conv2',strides=(1,1),padding='same')
        self.block2_bn = tf.keras.layers.BatchNormalization()
        self.block2_act = tf.keras.layers.ReLU()
        self.block2_pool=tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,name='block2_pool')
        
        self.block3_conv1=tf.keras.layers.Conv2D(64,(3,3),name='block2_conv1',strides=(1,1),padding='same')
        self.block3_conv2=tf.keras.layers.Conv2D(64,(3,3),name='block2_conv2',strides=(1,1),padding='same')
        self.block3_bn = tf.keras.layers.BatchNormalization()
        self.block3_act=tf.keras.layers.ReLU()
        self.block3_dp=tf.keras.layers.Dropout(0.5)

    def call(self,x):
        x=tf.image.rgb_to_grayscale(x,name=None)
    
        z1=self.block1(x)
        z1=self.block1_conv1(z1)
        z1=self.block1_conv2(z1)
        z1=self.block1_bn(z1)
        z1=self.block1_act(z1)
        z1_pool=self.block1_pool(z1)

        z2=self.block2_conv1(z1_pool)
        z2=self.block2_conv2(z2)
        z2=self.block2_bn(z2)
        z2=self.block2_act(z2)
        z2_pool=self.block1_pool(z2)

        z3=self.block3_conv1(z2_pool)
        z3=self.block3_conv2(z3)
        z3=self.block3_bn(z3)
        z3_pool=self.block3_act(z3)
        z3_dropout=self.block3_dp(z3_pool)
        return z1,z2,z3_dropout 

class Decoder(Model):
    def __init__(self):
        super().__init__()
        self.block4_up=tf.keras.layers.UpSampling2D(size=(2,2))
        self.block4_conv1=tf.keras.layers.Conv2D(32,(3,3),name='block4_conv1',strides=(1,1),padding='same')
        self.block4_conv2=tf.keras.layers.Conv2D(32,(3,3),name='block4_conv2',strides=(1,1),padding='same')
        self.block4_bn=tf.keras.layers.BatchNormalization()
        self.block4_act=tf.keras.layers.ReLU()

        self.block5_up=tf.keras.layers.UpSampling2D(size=(2,2))
        self.block5_conv1=tf.keras.layers.Conv2D(16,(3,3),name='block5_conv1',strides=(1,1),padding='same')
        self.block5_conv2=tf.keras.layers.Conv2D(16,(3,3),name='block5_conv1',strides=(1,1),padding='same')
        self.block5_bn=tf.keras.layers.BatchNormalization()
        self.block5_act=tf.keras.layers.ReLU()
        self.block5_conv3=tf.keras.layers.Conv2D(3,(3,3),name='block5_conv1',strides=(1,1),padding='same')
        
    

    def call(self,z1,z2,z3_dropout):
        z4_up=self.block4_up(z3_dropout)
        z4=self.block4_conv1(z4_up)
        #Encoderでmaxpool層前のfeaturemapをCropしてUp-Convolutionする時concatenationします
        z4=tf.keras.layers.concatenate([z2,z4],axis=3)
        z4=self.block4_conv2(z4)
        z4=self.block5_bn(z4)
        z4=self.block5_act(z4)

        z5_up=self.block5_up(z4)
        z5=self.block5_conv1(z5_up)
        z5=tf.keras.layers.concatenate([z1,z5],axis=3)
        z5=self.block5_conv2(z5)
        z5=self.block4_bn(z5)
        z5=self.block5_act
        y=self.block5_conv3(z5)

        return y

model=Unet()

#画像を出すメソッド
def generate_image(model,epoch,test_input):
    predictions=model(test_input,training=False)

    fig=plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i,:,:,0]*127.5+127.5,cmap='gray')
        plt.axis('off')
    #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def train(dataset,epochs):
    for epoch in range(epochs):
        start=time.time()
        for image_batch in dataset:

            model.train_step(image_batch)

train(train_dataset,EPOCHS)







