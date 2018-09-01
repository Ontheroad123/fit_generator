# fit_generator
keras中调用fit_gererator时，生成器的实现（此处训练集使用猫狗训练集）
fit_generator()函数就是解决训练集过大，无法一次性放入内容，每个batch的数据都从磁盘上获取,keras官网给的demo过于简单，这里自己实现整个过程。

1、将训练图片的绝对路径写入到文件中（也可以放到List中，这样读写的更快）
2、以猫狗数据集的数据格式为例，每次读取文件一行，然后通过地址读取图片，生成lable，注意这里lable必须是one-hot编码，不然报错：

ValueError: Error when checking target: expected activation_6 to have shape (2,) but got array with shape (1,)
3、设定batch_size，每次读取batch_size大小的图片之后，将图片和标签返回给模型
自己在当前目录下创建三个如下文件:
path.txt是整个猫狗数据集文件，
path_train.txt是训练集文件，
path_test.txt是测试集文件
test.py是使用fit_generator()训练（VGG 2D模型）
fit_generator-vgg3D.py是使用使用四维猫狗训练集训练VGG-3D模型
