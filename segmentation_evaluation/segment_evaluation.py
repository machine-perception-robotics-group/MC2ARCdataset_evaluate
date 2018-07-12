# -*- coding: utf-8 -*

import numpy as np
import cv2
from glob import glob
from os import path
import sys

# PLEASE CHANGE ##############################################

# Grand truth image's dir path
gt_path     = "./label/"

# Result image's dir path
result_path = "./results/"

# END ########################################################


FILE_gt  = glob(gt_path + "*.png")
FILE_pre = glob(result_path + "*.png")

FILE_gt.sort()
FILE_pre.sort()

gt_num  = len(FILE_gt)
pre_num = len(FILE_pre)

if gt_num != pre_num:
    print("[Error] Each(GT & result) number of files is different")
    print("GT    :" + str(gt_num))
    print("result:" + str(pre_num))
    sys.exit()
else:
    print("number of result images: " + str(gt_num))

for file_path in FILE_pre:
    path_items = file_path.split('/')
    file_name, ext = path.splitext(path_items[-1])

    gt_img  = cv2.imread(    gt_path + file_name + "_s" + ext)
    pre_img = cv2.imread(result_path + file_name        + ext)

    if gt_img.shape[0] != pre_img.shape[0] or gt_img.shape[1] != pre_img.shape[1]:
        print("[Error] Image size is different:" + file_path)
        sys.exit()
    else:
        SIZE = gt_img.shape[0] * pre_img.shape[1]

array = np.ones([900, 3])
array[:, :] = 999
index = 0
print("Extracting RGB Color")

i = 0
for file_path in FILE_pre:
    i = i+1
    path_items = file_path.split('/')
    file_name, ext = path.splitext(path_items[-1])

    gt_img  = cv2.imread(    gt_path + file_name + "_s" + ext)
    pre_img = cv2.imread(result_path + file_name        + ext)

    b = gt_img[:, :, 0]
    g = gt_img[:, :, 1]
    r = gt_img[:, :, 2]

    for height in xrange(gt_img.shape[0]):
        for width in xrange(gt_img.shape[1]):
            b = gt_img[height, width, 0]
            g = gt_img[height, width, 1]
            r = gt_img[height, width, 2]

            same = 0
            for counter in xrange(index):
                if array[counter, 0] == r and array[counter, 1] == g and array[counter, 2] == b:
                    same += 1

            if same == 0:
                np.vstack((array, (r, g, b)))
                array[index, 0] = r
                array[index, 1] = g
                array[index, 2] = b
                index += 1

    print(str(i) + '/' + str(gt_num))
for class_num in range(index):
    print(array[class_num])
print("Included RGB Color is " + str(index))
matrix = np.zeros([index, index]).astype(np.float32)

i = 0
for file_path in FILE_pre:
    i = i+1
    path_items = file_path.split('/')
    file_name, ext = path.splitext(path_items[-1])

    gt_img  = cv2.imread(    gt_path + file_name + "_s" + ext)
    pre_img = cv2.imread(result_path + file_name        + ext)

    print(i)

    for height in xrange(gt_img.shape[0]):
        for width in xrange(gt_img.shape[1]):
            for Input in range(index):
                if all(gt_img[height, width, (2, 1, 0)] == array[Input, (0, 1, 2)]):
                    for Output in range(index):
                        if all(pre_img[height, width, (2, 1, 0)] == array[Output, (0, 1, 2)]):
                            matrix[Input, Output] += 1.0

print(gt_img.shape[0] * gt_img.shape[1])
print(matrix)
print(index)

G_pixel = 0
for G_num in range(index):
    G_pixel += matrix[G_num, G_num]
GA = G_pixel / matrix.sum()
print(G_pixel)
print("Global Accuracy: " + str(GA * 100) + "%")

ca = 0
for C_num in range(index):
    ca += matrix[C_num, C_num] / matrix[C_num, :].sum()
CA = ca / index
print("Class Accuracy : " + str(CA * 100) + "%")

mi = 0
for M_num in range(index):
    mi += matrix[M_num, M_num] / (matrix[M_num, :].sum() + matrix[:, M_num].sum() - matrix[M_num, M_num])
MI = mi / index
print("Mean IoU       : " + str(MI * 100) + "%")
