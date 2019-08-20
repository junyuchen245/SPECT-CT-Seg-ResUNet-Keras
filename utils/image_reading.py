import os
import numpy as np
import warnings
#import SimpleITK as sitk
import cv2
from scipy import misc
from scipy import ndimage


def load_image_from_folder(folder_path, new_size, HE=False, Truc=False, Aug=False):
    """loads images in the folder_path and returns a ndarray and threshold the label image"""

    image_list = []
    label_list = []
    #counter = 0
    for image_name in os.listdir(folder_path):
        image_original = np.load(folder_path + image_name)
        image_original = image_original['a']
        #print(image_original.shape)
        #counter = counter + 1
        #print image_name, counter
        image_spect = image_original[:, 0:len(image_original)]
        image_ct = image_original[:,len(image_original):len(image_original)*2]
        label = image_original[:,len(image_original)*2:len(image_original)*3]
        #image_ct = cv2.resize(image_ct, new_size)
        #image_spect = cv2.resize(image_spect, new_size)
        #label = cv2.resize(label, new_size)
        #activate below for binary-class segmentation
        #super_threshold_indices = label != 0
        #label[super_threshold_indices] = 255
        #label = label / 255.0

        if HE == True:
            image_ct = cv2.equalizeHist(image_ct)
            image_spect = cv2.equalizeHist(image_spect)
        elif Truc == True:
            clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8,8))
            image_spect = clahe.apply(image_spect)
            image_ct = clahe.apply(image_ct)
            #ret, image = cv2.threshold(image,200,255,cv2.THRESH_TRUNC)
        else:
            image_spect = image_spect
            image_ct = image_ct

#image augmentation method in the FusionNet paper
        if Aug == True:
            '''SPECT'''
            imageSPECT_aug_1 = ndimage.rotate(image_spect, -90)
            imageSPECT_aug_2 = np.flipud(imageSPECT_aug_1)
            imageSPECT_aug_3 = ndimage.rotate(image_spect, -180)
            imageSPECT_aug_4 = np.flipud(imageSPECT_aug_3)
            imageSPECT_aug_5 = ndimage.rotate(image_spect, -270)
            imageSPECT_aug_6 = np.flipud(imageSPECT_aug_5)
            imageSPECT_aug_7 = np.flipud(image_spect)

            '''CT'''
            imageCT_aug_1 = ndimage.rotate(image_ct, -90)
            imageCT_aug_2 = np.flipud(imageCT_aug_1)
            imageCT_aug_3 = ndimage.rotate(image_ct, -180)
            imageCT_aug_4 = np.flipud(imageCT_aug_3)
            imageCT_aug_5 = ndimage.rotate(image_ct, -270)
            imageCT_aug_6 = np.flipud(imageCT_aug_5)
            imageCT_aug_7 = np.flipud(image_ct)

            '''label'''
            label_aug_1 = ndimage.rotate(label, -90)
            label_aug_1 = label_aug_1.astype(int)
            label_aug_2 = np.flipud(label_aug_1)
            label_aug_2 = label_aug_2.astype(int)
            label_aug_3 = ndimage.rotate(label, -180)
            label_aug_3 = label_aug_3.astype(int)
            label_aug_4 = np.flipud(label_aug_3)
            label_aug_4 = label_aug_4.astype(int)
            label_aug_5 = ndimage.rotate(label, -270)
            label_aug_5 = label_aug_5.astype(int)
            label_aug_6 = np.flipud(label_aug_5)
            label_aug_6 = label_aug_6.astype(int)
            label_aug_7 = np.flipud(label)
            label_aug_7 = label_aug_7.astype(int)


            image_all_0 = np.concatenate((image_ct,image_spect),axis=1)
            image_all_1 = np.concatenate((imageCT_aug_1, imageSPECT_aug_1), axis=1)
            image_all_2 = np.concatenate((imageCT_aug_2, imageSPECT_aug_2), axis=1)
            image_all_3 = np.concatenate((imageCT_aug_3, imageSPECT_aug_3), axis=1)
            image_all_4 = np.concatenate((imageCT_aug_4, imageSPECT_aug_4), axis=1)
            image_all_5 = np.concatenate((imageCT_aug_5, imageSPECT_aug_5), axis=1)
            image_all_6 = np.concatenate((imageCT_aug_6, imageSPECT_aug_6), axis=1)
            image_all_7 = np.concatenate((imageCT_aug_7, imageSPECT_aug_7), axis=1)

            image_list.append(image_all_0)
            image_list.append(image_all_1)
            image_list.append(image_all_2)
            image_list.append(image_all_3)
            image_list.append(image_all_4)
            image_list.append(image_all_5)
            image_list.append(image_all_6)
            image_list.append(image_all_7)

            label_list.append(label)
            label_list.append(label_aug_1)
            label_list.append(label_aug_2)
            label_list.append(label_aug_3)
            label_list.append(label_aug_4)
            label_list.append(label_aug_5)
            label_list.append(label_aug_6)
            label_list.append(label_aug_7)
        else:
            image_all = np.concatenate((image_ct, image_spect), axis=1)
            image_list.append(image_all)
            label_list.append(label)

    image_array = np.asarray(image_list)
    label_array = np.asarray(label_list)

    return image_array, label_array


def load_test_from_folder(folder_path, new_size, HE=False, Truc=False, Aug=False):
    """loads images in the folder_path and returns a ndarray and threshold the label image"""

    image_list = []
    #counter = 0
    for image_name in os.listdir(folder_path):
        image_original = np.load(folder_path + image_name)
        image_original = image_original['a']
        #counter = counter + 1
        #print image_name, counter
        image_ct = image_original[:, 0:len(image_original)]
        image_spect = image_original[:,len(image_original):len(image_original)*2]

        image_all = np.concatenate((image_ct, image_spect), axis=1)
        image_list.append(image_all)


    image_array = np.asarray(image_list)

    return image_array