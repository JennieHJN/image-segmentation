'''
    Func:制作训练集
'''

import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import cv2

ct_name = ".nii"
mask_name = ".nii"

ct_path = '../../LITS2017/valid/CT'
seg_path = '../../LITS2017/valid/seg'
png_path = './png/'

outputImg_path = "../data/validImage"
outputMask_path = "../data/validMask"

if not os.path.exists(outputImg_path):
    os.makedirs(outputImg_path)
if not os.path.exists(outputMask_path):
    os.makedirs(outputMask_path)

def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files

def crop_ceter(img, croph, cropw):
    #for n_slice in range(img.shape[0]):
    height, width = img[0].shape
    starth = height//2 - (croph//2)
    startw = width//2 - (cropw//2)
    return img[:, starth:starth+croph, startw:startw+cropw]

if __name__ == "__main__":

    for index, file in enumerate(tqdm(os.listdir(ct_path))):


        ct_src = sitk.ReadImage(os.path.join(ct_path, file), sitk.sitkInt16)
        mask = sitk.ReadImage(os.path.join(seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        ct_array = sitk.GetArrayFromImage(ct_src)
        mask_array = sitk.GetArrayFromImage(mask)



        ct_array[ct_array > 200] = 200
        ct_array[ct_array < -200] = -200

        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200


        z = np.any(mask_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        print('cs',start_slice,end_slice,file)
        start_slice = max(0, start_slice - 1)
        end_slice = min(mask_array.shape[0] - 1, end_slice + 2)

        ct_crop = ct_array[start_slice:end_slice, :, :]
        mask_crop = mask_array[start_slice+1:end_slice-1, :, :]

        ct_crop = ct_crop[:,32:480,32:480]
        mask_crop = mask_crop[:,32:480,32:480]

        print('ct_crop.shape',ct_crop.shape)

        if int(np.sum(mask_crop))!=0:
            for n_slice in range(mask_crop.shape[0]):
                maskImg = mask_crop[n_slice, :, :]
                ctImageArray = np.zeros((ct_crop.shape[1], ct_crop.shape[2], 3), np.float)
                ctImageArray[:, :, 0] = ct_crop[n_slice , :, :]
                ctImageArray[:, :, 1] = ct_crop[n_slice + 1, :, :]
                ctImageArray[:, :, 2] = ct_crop[n_slice + 2, :, :]

                imagepath = outputImg_path + "/" + str(index+1) + "_" + str(n_slice) + ".npy"
                maskpath = outputMask_path + "/" + str(index+1) + "_" + str(n_slice) + ".npy"
                    
                np.save(imagepath, ctImageArray)
                np.save(maskpath, maskImg)
            continue
    print("Done！")
