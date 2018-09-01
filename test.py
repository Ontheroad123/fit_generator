'''
2018-8-21
fit_generator,设置多线程来训练，但是还是有一个缺点，多线程中每次还是训练一张图片，并不是一次训练batch_size张图片
heq
'''
import os
from PIL import Image
from keras.optimizers import SGD
import numpy as np
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


# 读取样本名称，然后根据样本名称去读取数据
def endwith(s, *endstring):
    resultArray = map(s.endswith, endstring)
    if True in resultArray:
        return True
    else:
        return False


# 将训练集图片地址全部写入txt文件中
def write_imgpath():
    path = "./train"
    fp = open('path.txt', 'w')
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        for sub_file in os.listdir(file_path):
            if endwith(sub_file, 'jpg'):
                image_path = (os.path.join(file_path, sub_file))
                fp.write(image_path + "\n")

    fp.close()

#划分训练集和测试集
def write_test_imgpath():
    write_imgpath()
    path1 = 'path.txt'
    fp1 = open('path_train.tst', 'w')
    fp2 = open('path_test.txt', 'w')
    count = 0
    with open(path1) as f:
        for line in f:
            count += 1
            line = line.replace('\n', '')
            if count <= 2500:
                fp2.write(line + "\n")
            elif count >= 22501:
                fp2.write(line + "\n")
            else:
                fp1.write(line + "\n")


write_test_imgpath()

# 传入batch_size和需要打开的文件地址
def generator_data(batch_size, path):
    list_x = []
    list_y = []
    count = 0
    while True:

        with open(path) as f:
            for line in f:
                line = line.replace('\n', '')
                x, y = process_line(line)
                list_x.append(x)
                list_y.append(y)
                count += 1
                if count >= batch_size:
                    yield (np.array(list_x), np.array(list_y))
                    count = 0
                    list_x = []
                    list_y = []

# 读取图片和标签
def process_line(line):
    img = Image.open(line)

    img = img.resize((255, 255))

    str = line.split('.')
    str1 = str[0].split('/')
    length = len(str1)
    if str1[length - 1] == 'dog':
        label = 1
    else:
        label = 0
    emable = to_categorical(label, 2)  # one-hot编码
    img = np.array(img)
    return img, emable

def VGG_16():
    model = Sequential()
    model.add(Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv1',
                     dim_ordering='tf',
                     input_shape=(255, 255, 3)))
    model.add(Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv1'))
    model.add(Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv1'))
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv2'))
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv1'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv2'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv1'))
    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv2'))

    model.add(Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid', name='predictions'))
    print(model.summary())
    return model


if __name__ == '__main__':
    model = VGG_16()
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    model.fit_generator(generator_data(32, "./path_train.txt"),
                        steps_per_epoch=100,#每次迭代训练100张图片
                        epochs=10,
                        workers=10,
                        use_multiprocessing=True,
                        callbacks=[earlyStopping]
                        )
    loss, accuracy = model.evaluate_generator(generator_data(32, "./path_test.txt"), steps=10)
    print("loss is :", loss)
    print("accuracy is :", accuracy)



