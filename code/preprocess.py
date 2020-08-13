from PIL import Image
from scipy.io import savemat
import numpy as np
import struct
import os

# 处理.idx类型的mnist数据源
def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    # 上述方法得到(60000, 784)维的数据和（60000）维的标签
    tmp_idx = list(range(10))
    # 不要用fromkeys方法构建字典列表，详细见https://www.lfhacks.com/tech/Python-empty-dictionary
    lb_dic = dict([(k,[]) for k in tmp_idx]) 
    for idx in range(len(images)):
        org_img = images[idx].reshape(28,28)
        im = Image.fromarray(org_img)
        resized_im = im.resize((16, 16))
        new_img = np.asarray(resized_im)
        # 注意list每阶的维度一样才能表示成大张量
        # 详细见https://stackoverflow.com/questions/52097119/incorrect-shape-of-2d-numpy-array
        if len(lb_dic[labels[idx]]) < 90:
            lb_dic[labels[idx]].append(new_img)
    result=np.array([np.array(x) for x in lb_dic.values()])
    result=result.swapaxes(2,0).swapaxes(3,1)
    print(result.shape)
    mat_dic = {}
    mat_dic['Data_group'] = result
    # return result
    savemat(os.path.join(path,kind+".mat"), mat_dic)

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
        new_img = np.asarray(resized_im)
        if len(lb_dic[labels[idx]]) < 800:
            lb_dic[labels[idx]].append(new_img)
    result=np.array([np.array(x) for x in lb_dic.values()])
    result=result.swapaxes(2,0).swapaxes(3,1) # 维度交换swapaxes，而不是转置(transpose)
    print(result.shape)
    # 下面三行为图片测试程序，可以自行查看交换后的效果
    # test_img = result[:,:,9,0]
    # im = Image.fromarray(test_img)
    # im.show()
    mat_dic = {}
    mat_dic['Data_group'] = result
    mat_dic['Data_test_group'] = result
    savemat(os.path.join(path,"test.mat"), mat_dic)
    # return result

if __name__ == "__main__":
    mat_dic = {}
    load_mnist_train("code\\samples")
    load_mnist_test("code\\samples")