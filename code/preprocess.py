from PIL import Image
from scipy.io import savemat
import numpy as np
import struct
import os

# 处理.idx类型的mnist数据源
def load_mnist_train(path, samples, test_size, kind='train'):
    """
    path: 训练集读取路径
    samples: 每类label的样本数目
    test_size: 测试集所占比例
    kind: 读取文件类型
    """
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    t = round(samples * test_size) # 测试集大小
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    # 上述方法得到(60000, 784)维的数据和（60000）维的标签
    tmp_idx = list(range(10))
    # 不要用fromkeys方法构建字典列表，详细见https://www.lfhacks.com/tech/Python-empty-dictionary
    lb_dic_train = dict([(k,[]) for k in tmp_idx]) # 每个label模型对应的训练集
    lb_dic_test = dict([(k,[]) for k in tmp_idx]) # 每个label模型对应的测试集
    for idx in range(len(images)):
        org_img = images[idx].reshape(28,28)
        im = Image.fromarray(org_img)
        resized_im = im.resize((16, 16)) 
        new_img = new_img = np.asarray(resized_im) / 255.0
        # 注意list每阶的维度一样才能表示成大张量
        # 详细见https://stackoverflow.com/questions/52097119/incorrect-shape-of-2d-numpy-array
        if len(lb_dic_train[labels[idx]]) < samples:
            lb_dic_train[labels[idx]].append(new_img)
    # train_test_split思想划分数据集
    for cla in range(len(lb_dic_train)):
        # print('修改之前样本数%s'%len(lb_dic_train[cla]))
        idxs = range(samples)
        idx = np.random.choice(idxs, size=t, replace=False)
        temp = np.array(lb_dic_train[cla])
        lb_dic_train[cla] = temp[list(set(idxs) - set(idx))]
        lb_dic_test[cla] = temp[idx]
        # print('修改之后训练样本数%s'%len(lb_dic_train[cla]))
        # print('修改之后训练test数%s'%len(lb_dic_test[cla]))

    train = np.array([np.array(x) for x in lb_dic_train.values()]) 
    test = np.array([np.array(x) for x in lb_dic_test.values()]) 
    train=train.swapaxes(2,0).swapaxes(3,1)
    test=test.swapaxes(2,0).swapaxes(3,1)
    # print(result.shape)
    mat_dic = {}
    mat_dic['Data_group'] = train
    # return result
    savemat(os.path.join(path,kind+".mat"), mat_dic)
    return train,test

def load_mnist_test(path, kind='t10k'):
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    # 上述方法得到(10000, 784)维的数据和（10000）维的标签
    tmp_idx = list(range(10))
    lb_dic = dict([(k,[]) for k in tmp_idx]) 
    for idx in range(len(images)):
        org_img = images[idx].reshape(28,28)
        im = Image.fromarray(org_img)
        resized_im = im.resize((16, 16)) 
        new_img = np.asarray(resized_im) / 255.0
        new_img=np.around(new_img, decimals=2)
        if len(lb_dic[labels[idx]]) < 800:
            lb_dic[labels[idx]].append(new_img)
    result=np.array([np.array(x) for x in lb_dic.values()])
    result=result.swapaxes(2,0).swapaxes(3,1) # 维度交换swapaxes，而不是转置(transpose)
    print(result.shape)
    # 下面三行为图片测试程序，可以自行查看交换后的效果
    # test_img = result[:,:,9,0]
    # im = Image.fromarray(test_img)
    # im.show()
    # mat_dic = {}
    # mat_dic['Data_group'] = result
    # mat_dic['Data_test_group'] = result
    # savemat(os.path.join(path,"test.mat"), mat_dic)
    return result

if __name__ == "__main__":
    path = "code//samples//"
    train_mat = {}
    train, each_test = load_mnist_train(path,900,0.3)
    train_mat['Data_group'] = each_test
    cpl_test = load_mnist_test(path)
    train_mat['Data_test_group'] = cpl_test
    savemat(os.path.join(path,"test.mat"), train_mat)