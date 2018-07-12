#!/usr/bin/env python
#coding:utf-8

import numpy as np
import os
from PIL import Image
import cv2
import json

##### PLEASE CHANGE ##############################
#segmentation_label dir
seg_dir = '../../../../ARCdataset_png/test/seg_class_number/'

#occlusion_segmentation_label dir
ocl_dir = '../../../../ARCdataset_png/test/seg_instance/'

#predict_tree_json path
json_path = '../../ol_mask_rcnn/result/json/mask_ocl.json'

##################################################

def main():
    f = open(json_path, 'r')
    json_datas = json.load(f)

    true_tree = 0

    #img & path 取得
    for i in range(0, len(json_datas)):
        print(str(i) + '枚目')
        #json --- info & tree
        json_data = json_datas[i]
        json_data_info = json_data['info']
        json_data_tree = json_data['tree']

        json_data_info = json_data_info[0]

        #label --- path
        seg_path = seg_dir + json_data_info['label']
        ocl_path = json_data_info['label']
        ocl_path = ocl_path.replace('.png', '')
        ocl_path = ocl_dir + ocl_path + '/'

        #segmentation_label
        seg = cv2.imread(seg_path, 0)

        true_mask = 0

        #mask
        for mask_i in range(0, len(json_data_tree)):
            tree = json_data_tree[mask_i]
            occlusion = tree['occlusion']
            bbox = tree['bbox']

            x_min = int(bbox[1])
            y_min = int(bbox[0])
            x_max = int(x_min + bbox[3])
            y_max = int(y_min + bbox[2])

            mask = seg[x_min:x_max, y_min:y_max]
            mask_where = np.where(mask != 0)

            #class_num 取得
            class_nums = []
            class_count = []

            for xy in range(0, len(mask_where[0])):
                class_num = mask[mask_where[0][xy], mask_where[1][xy]]

                if class_num not in class_nums:
                    class_nums.append(class_num)

            for class_num in class_nums:
                class_where = np.where(mask == class_num)
                class_count.append(len(class_where[0]))

            if len(class_nums) != 0:
                class_num = class_nums[np.argmax(class_count)]

                #ocl_seg_path
                ocl_file_name = json_data_info['label']
                ocl_dir_name = ocl_file_name.replace('.png', '/')
                ocl_file_name = ocl_file_name.replace('.png', '_' + str(class_num) + '.png')
                ocl_path = ocl_dir + ocl_dir_name + ocl_file_name

                if len(occlusion) != 0:
                    #Layer 2 ~ Layer N
                    ocl_seg = cv2.imread(ocl_path, 0)
                    ocl_seg[ocl_seg != 0] = 255

                    true_count = 0

                    for o_i in range(0, len(json_data_tree)):
                        ocl_data = json_data_tree[o_i]
                        ocl_mask_name = ocl_data['mask']

                        ocl_class_nums = []
                        ocl_class_count = []

                        if ocl_mask_name in occlusion:
                            ocl_bbox = ocl_data['bbox']

                            ocl_x_min = int(ocl_bbox[1])
                            ocl_y_min = int(ocl_bbox[0])
                            ocl_x_max = int(ocl_x_min + ocl_bbox[3])
                            ocl_y_max = int(ocl_y_min + ocl_bbox[2])

                            ocl_mask = seg[ocl_x_min:ocl_x_max, ocl_y_min:ocl_y_max]
                            ocl_mask_where = np.where(ocl_mask != 0)

                            for xy in range(0, len(ocl_mask_where[0])):
                                ocl_class_num = ocl_mask[ocl_mask_where[0][xy], ocl_mask_where[1][xy]]

                                if ocl_class_num not in ocl_class_nums:
                                    ocl_class_nums.append(ocl_class_num)

                            for ocl_class_num in ocl_class_nums:
                                ocl_class_where = np.where(ocl_mask == ocl_class_num)
                                ocl_class_count.append(len(ocl_class_where[0]))

                            if len(ocl_class_nums) != 0:
                                ocl_class_num = ocl_class_nums[np.argmax(ocl_class_count)]

                            #ocl_seg_path2
                            ocl_file_name2 = json_data_info['label']
                            ocl_file_name2 = ocl_file_name2.replace('.png', '_' + str(ocl_class_num) + '.png')
                            ocl_path2 = ocl_dir + ocl_dir_name + ocl_file_name2

                            ocl_seg2 = cv2.imread(ocl_path2, 0)
                            ocl_seg2[ocl_seg2 != 0] = 255

                            ocl_seg3 = ocl_seg - ocl_seg2
                            ocl_count1 = np.where(ocl_seg == 255)
                            ocl_count2 = np.where(ocl_seg3 == 255)

                            if len(ocl_count1[0]) > len(ocl_count2[0]):
                                true_count += 1

                            if true_count == len(occlusion):
                                true_mask += 1


                else:
                    #Layer 1
                    ocl_seg = cv2.imread(ocl_path, 0)
                    ocl_mask = ocl_seg[x_min:x_max, y_min:y_max]
                    ocl_where = np.where(ocl_mask == 255)

                    if len(ocl_where[0]) == 0:
                        true_mask += 1

            else:
                print('bbox is missed')

        print('True : ' + str(true_mask) + '/' + str(len(json_data_tree)))

        if true_mask == len(json_data_tree):
            true_tree += 1
    print('Accuracy : ' + str(true_tree) + '/' + str(len(json_datas)))


if __name__ == '__main__':
    main()
