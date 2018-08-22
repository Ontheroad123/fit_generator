'''
2018-8-21
fit_generator
heq
'''
import os
from keras.optimizers import SGD
import cv2
import numpy as np
from keras.layers import Dense, Activation, Conv2D, ZeroPadding2D,MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# 读取样本名称，然后根据样本名称去读取数据
def endwith(s,*endstring):
   resultArray = map(s.endswith,endstring)
   if True in resultArray:
       return True
   else:
       return False

#将训练集图片地址全部写入txt文件中
def write_train_imgpath():
    path="/home/hq/桌面/cat_dog/c-d-data/train"
    fp = open('path_train.txt', 'w')
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        for sub_file in os.listdir(file_path):
            if endwith(sub_file, 'jpg'):
                image_path = (os.path.join(file_path, sub_file))
                fp.write(image_path+"\n")
    fp.close()
#write_train_imgpath()
def write_test_imgpath():
    path1='path_train.txt'
    fp2 = open('path_test.txt','w')
    count=0
    with open(path1) as f:
        for line in f:
            count+=1
            line = line.replace('\n', '')
            if count<=2500:
                fp2.write(line+"\n")
            elif count>=22501:
                fp2.write(line+"\n")


#这里偷懒将训练集一部分地址直接放到path_test.txt文件中了
'''def write_test_imgpath():
    path = "/home/hq/桌面/cat_dog/c-d-data/test"
    fp = open('path_test.txt', 'w')
    for file in os.listdir(path):
        image_path = (os.path.join(path, file))
        fp.write(image_path + "\n")
    fp.close()
'''

#传入batch_size和需要打开的文件地址
def generator_data(batch_size,path):
    list_x=[]
    list_y=[]
    count=0
    while True:

        with open(path) as f:
            for line in f:
                line = line.replace('\n', '')
                x,y = process_line(line)
                list_x.append(x)
                list_y.append(y)
                count+=1
                if count>=batch_size:
                    yield (np.array(list_x),np.array(list_y))
                    count=0
                    list_x=[]
                    list_y=[]
#读取图片和标签
def process_line(line):

    img = cv2.imread(line)

    img = cv2.resize(img, (255, 255), interpolation=cv2.INTER_CUBIC)
    str=line.split('.')
    str1=str[0].split('/')
    length=len(str1)
    if str1[length-1]=='dog':
        label=1
    else:
        label=0
    emable=to_categorical(label, 2)#one-hot编码
    return img, emable

def VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(255,255,3)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model

model = VGG_16()
sgd = SGD(lr=0.000001,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(optimizer=sgd,  loss='squared_hinge',  metrics=['accuracy'])
#epoch=n*steps_per_epoch*batch_size,自己理解的这样，在设置batch_size和steps的时候注意
'''for x,y in generator_data(10,"/home/hq/桌面/cat_dog/path.txt"):
    print(y)
    break'''
model.fit_generator(generator_data(32, "/home/hq/桌面/cat_dog/path_train.txt"),
                    steps_per_epoch=20,
                    epochs=10
                    )
loss ,accuracy= model.evaluate_generator(generator_data(32, "/home/hq/桌面/cat_dog/path_test.txt"), steps=20)
print("loss is :",loss)
print("accuracy is :",accuracy)



