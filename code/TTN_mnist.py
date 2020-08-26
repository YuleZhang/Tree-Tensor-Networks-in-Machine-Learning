#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import product
import numpy as np
import pickle

from tree_tensor_network_mnist import TreeTensorNetwork
from utilities_mnist import load_train_data
from utilities_mnist import load_test_data



data_folder = "./code/samples/"
n_epochs = 3

bond_data = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] # 待分解张量的阶数
# bond_data = [9, 10, 9, 10, 10, 9,9,9,9,8]
bond_inner = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] # 辅助指标的维数
# bond_inner = [9, 10, 9,10, 10, 9, 9, 9, 9, 8]
bond_label = 2

n_class = 2
n_train_single = 10
n_train_each = n_train_single * 9
n_train = n_train_each * n_class
print(n_train)

n_test_each = 80
n_test = n_test_each * 10

layer_units = [16, 8, 4, 2, 1]

# build tensor network---------------------------------------------------
print("building tensor network")
output = open('10_class_model.pkl', 'wb')

ttn = [0 for col in range(10)]
acc_train = [0 for col in range(10)]
acc_test1 = [0 for col in range(10)]
acc_test2 = [0 for col in range(10)]

# train_images,train_labels=load_mnist_train("code\\samples")

for i in range(10):
    print("bond_data:", bond_data[i])
    print("bond_inner:", bond_inner[i])
    # 每次读取不同的标签
    data, labels = load_train_data(
        data_folder + "train.mat", n_train, n_train_each, bond_label, bond_data[i], i)
    data_test, labels_test = load_train_data(
        data_folder + "test.mat", n_test_each * 2, n_test_each, bond_label, bond_data[i], i)

    ttn[i] = TreeTensorNetwork(
        data, labels, bond_data[i], bond_inner[i], bond_label, layer_units)

    # Training
    print("Training number", i, "th binary classifier:")
    acc_train[i] = ttn[i].train(n_epochs)
    print("training inner product:", acc_train[i])

    # Testing for each classifier
    print("Testing number", i, "th binary classifier:")
    acc_test1[i], acc_test2[i] = ttn[i].test(data_test, labels_test)
    print("testing accuracy1 {0}, testing accuarcy2 {1}".format(acc_test1[i], acc_test2[i]))

    pickle.dump(ttn, output)
output.close()

# Testing on 10 classes
output_vector = np.zeros((n_test, 10), dtype=np.float64)
output_label = np.zeros((n_test), dtype=np.float64)
count = 0
for i in range(10):
    test_tensor = load_test_data(
        data_folder + "test.mat", n_test, n_test_each, bond_data[i])
    output_vector[:, i] = ttn[i].outputvalue(test_tensor, n_test)

for i, j in product(range(10), range(n_test_each)):
    output_label[i * n_test_each +
                 j] = np.argmax(output_vector[i * n_test_each + j, :])
    if output_label[i * n_test_each + j] == i:
        count = count + 1

print("testing accuracy on 10 classes:", count / n_test)
