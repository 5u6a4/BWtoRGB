import os 
import cv2
import glob  

LABEL_DIR='images'
TRAIN_PATH=f'{LABEL_DIR}/train'
TEST_PATH=f'{LABEL_DIR}/test'

INPUT_DIR='images_BW'
TRAIN_BW_PATH=f'{INPUT_DIR}/train'
TRAIN_BW_PATH_apartment=f'{TRAIN_BW_PATH}/apartment'
TRAIN_BW_PATH_church=f'{TRAIN_BW_PATH}/church'
TRAIN_BW_PATH_garage=f'{TRAIN_BW_PATH}/garage'
TRAIN_BW_PATH_house=f'{TRAIN_BW_PATH}/house'
TRAIN_BW_PATH_insdustrial=f'{TRAIN_BW_PATH}/industrial'
TRAIN_BW_PATH_office=f'{TRAIN_BW_PATH}/office'
TRAIN_BW_PATH_retail=f'{TRAIN_BW_PATH}/retail'
TRAIN_BW_PATH_roof=f'{TRAIN_BW_PATH}/roof'


TEST_BW_PATH=f'{INPUT_DIR}/test'
TEST_BW_PATH_apartment=f'{TEST_BW_PATH}/apartment'
TEST_BW_PATH_church=f'{TEST_BW_PATH}/church'
TEST_BW_PATH_garage=f'{TEST_BW_PATH}/garage'
TEST_BW_PATH_house=f'{TEST_BW_PATH}/house'
TEST_BW_PATH_insdustrial=f'{TEST_BW_PATH}/industrial'
TEST_BW_PATH_office=f'{TEST_BW_PATH}/office'
TEST_BW_PATH_retail=f'{TEST_BW_PATH}/retail'
TEST_BW_PATH_roof=f'{TEST_BW_PATH}/roof'

if  not os.path.exists(INPUT_DIR):
    os.mkdir(INPUT_DIR)
    os.mkdir(TRAIN_BW_PATH)
    os.mkdir(TEST_BW_PATH)

    os.mkdir(TRAIN_BW_PATH_apartment)
    os.mkdir(TRAIN_BW_PATH_church)
    os.mkdir(TRAIN_BW_PATH_garage)
    os.mkdir(TRAIN_BW_PATH_house)
    os.mkdir(TRAIN_BW_PATH_insdustrial)
    os.mkdir(TRAIN_BW_PATH_office)
    os.mkdir(TRAIN_BW_PATH_retail)
    os.mkdir(TRAIN_BW_PATH_roof)

    os.mkdir(TEST_BW_PATH_apartment)
    os.mkdir(TEST_BW_PATH_church)
    os.mkdir(TEST_BW_PATH_garage)
    os.mkdir(TEST_BW_PATH_house)
    os.mkdir(TEST_BW_PATH_insdustrial)
    os.mkdir(TEST_BW_PATH_office)
    os.mkdir(TEST_BW_PATH_retail)
    os.mkdir(TEST_BW_PATH_roof)


dir_list_train=os.listdir(f'{TRAIN_PATH}')
dir_list_apartment=os.listdir(f'{TRAIN_PATH}/apartment')
dir_list_a=[]
print(dir_list_train)
print(dir_list_apartment[2])
print(len(dir_list_apartment))
for i in dir_list_train:
    dir_list_a=[]
    count=0
    dir=os.path.join(f'{TRAIN_PATH}',i)
    dir_list=os.listdir(dir)
    for j in dir_list:
        
        dir_list_a.append(os.path.join(os.path.join(f'{TRAIN_PATH}',f'{i}'),j))
        im_gray=cv2.imread(f'{dir_list_a[count]}',0)
        count+=1
        # filename=dir_list_apartment[i]
        if i=='apartment': 
            filename=os.path.join(f'{TRAIN_BW_PATH_apartment}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='church':
            filename=os.path.join(f'{TRAIN_BW_PATH_church}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='garage':
            filename=os.path.join(f'{TRAIN_BW_PATH_garage}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='house':
            filename=os.path.join(f'{TRAIN_BW_PATH_house}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='industrial':
            filename=os.path.join(f'{TRAIN_BW_PATH_insdustrial}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='office':
            filename=os.path.join(f'{TRAIN_BW_PATH_office}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='retail':
            filename=os.path.join(f'{TRAIN_BW_PATH_retail}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='roof':
            filename=os.path.join(f'{TRAIN_BW_PATH_roof}',j)
            cv2.imwrite(f'{filename}', im_gray)
        # if count>2:
        #     break


# for i in range(len(dir_list_apartment)):
#     dir_list_a.append(os.path.join(f'{TRAIN_PATH}/apartment',dir_list_apartment[i]))
#     im_gray=cv2.imread(f'{dir_list_a[i]}',0)
#     #print(type(im_gray))
#     filename=dir_list_apartment[i]
#     filename=os.path.join(f'{TRAIN_BW_PATH_apartment}',dir_list_apartment[i])
#     cv2.imwrite(f'{filename}', im_gray)
    #cv2.imwrite(TRAIN_BW_PATH)
    # name=f'{filename}'
    # print(name)

dir_list_test=os.listdir(f'{TEST_PATH}')
print(len(dir_list_apartment))
for i in dir_list_test:
    dir_list_a=[]
    count=0
    dir=os.path.join(f'{TEST_PATH}',i)
    dir_list=os.listdir(dir)
    for j in dir_list:
        
        dir_list_a.append(os.path.join(os.path.join(f'{TEST_PATH}',f'{i}'),j))
        im_gray=cv2.imread(f'{dir_list_a[count]}',0)
        count+=1
        # filename=dir_list_apartment[i]
        if i=='apartment': 
            filename=os.path.join(f'{TEST_BW_PATH_apartment}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='church':
            filename=os.path.join(f'{TEST_BW_PATH_church}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='garage':
            filename=os.path.join(f'{TEST_BW_PATH_garage}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='house':
            filename=os.path.join(f'{TEST_BW_PATH_house}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='industrial':
            filename=os.path.join(f'{TEST_BW_PATH_insdustrial}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='office':
            filename=os.path.join(f'{TEST_BW_PATH_office}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='retail':
            filename=os.path.join(f'{TEST_BW_PATH_retail}',j)
            cv2.imwrite(f'{filename}', im_gray)
        if i=='roof':
            filename=os.path.join(f'{TEST_BW_PATH_roof}',j)
            cv2.imwrite(f'{filename}', im_gray)
