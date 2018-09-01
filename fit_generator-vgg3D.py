'''
2018-8-21
fit_generator
heq
'''
import os
from PIL import ImageFilter,Image
from keras.optimizers import SGD
import numpy as np
from keras.layers import Conv3D,MaxPooling3D,Flatten
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# 读取样本名称，然后根据样本名称去读取数据
def endwith(s,*endstring):
   resultArray = map(s.endswith,endstring)
   if True in resultArray:
       return True
   else:
       return False

#将训练集图片地址全部写入txt文件中
def write_imgpath():
    path="./train"
    fp = open('path.txt', 'w')
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        for sub_file in os.listdir(file_path):
            if endwith(sub_file, 'jpg'):
                image_path = (os.path.join(file_path, sub_file))
                fp.write(image_path+"\n")
               
    fp.close()
    
#write_train_imgpath()
def write_test_imgpath():
    write_imgpath()
    path1='path.txt'
    fp1 = open('path_train.tst','w')
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
            else:
                fp1.write(line+"\n")
write_test_imgpath()

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
class GaussianBlur(ImageFilter.Filter):
    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:

            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def process_line(line):

    img = Image.open(line)

    img = img.resize((255, 255))

    str=line.split('.')
    str1=str[0].split('/')
    length=len(str1)
    if str1[length-1]=='dog':
        label=1
    else:
        label=0
    emable=to_categorical(label, 2)#one-hot编码

    arr = np.asarray(img, dtype="float32")
    image1 = img.filter(GaussianBlur(radius=1))
    arr1 = np.asarray(image1, dtype="float32")
    image2 = img.filter(GaussianBlur(radius=3))
    arr2 = np.asarray(image2, dtype="float32")
    image3 = img.filter(GaussianBlur(radius=5))
    arr3 = np.asarray(image3, dtype="float32")
    #合成四维矩阵
    new = np.empty((255, 255, 3, 4), dtype="float32")
    new[:, :, :, 0] = arr
    new[:, :, :, 1] = arr1
    new[:, :, :, 2] = arr2
    new[:, :, :, 3] = arr3

    return new ,emable

def VGG_16():
    model = Sequential()
    model.add(Conv3D(64, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv1',
                     dim_ordering='tf',
                     input_shape=(255, 255, 3, 4)))
    model.add(Conv3D(64, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv2'))
    model.add(MaxPooling3D(
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        name='block1_pool',
        padding='same'
    ))
    model.add(Conv3D(128, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv1'))
    model.add(Conv3D(128, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv2'))

    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2),  # padding='same',
                           name='block2_pool'
                           ))

    model.add(Conv3D(256, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv1'))
    model.add(Conv3D(256, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv2'))
    model.add(Conv3D(256, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv3'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='block3_pool'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv1'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv2'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv3'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='block4_pool'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv1'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv2'))

    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv3'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='block5_pool'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))

    model.add(Dense(4096, activation='relu', name='fc2'))

    model.add(Dense(2, activation='sigmoid', name='predictions'))
    print(model.summary())

    return model


if __name__ == '__main__':
    model = VGG_16()
    sgd = SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(optimizer=sgd,  loss='binary_crossentropy',  metrics=['accuracy'])
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    model.fit_generator(generator_data(32, "./path_train.txt"),
                    steps_per_epoch=20,
                    epochs=10,
                    callbacks=[earlyStopping]
                    )
    loss ,accuracy= model.evaluate_generator(generator_data(32, "./path_test.txt"), steps=5)
    print("loss is :",loss)
    print("accuracy is :",accuracy)


