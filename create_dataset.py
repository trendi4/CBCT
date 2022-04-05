import glob
import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import cv2


def create_folder(name): 
    if os.path.exists(name):
        shutil.rmtree(name)
    os.makedirs(name)

def to_image(image, slope=0.0641, intercept=-1200, window_length=100, window_center=40):
    #calculate limitis to clip the image
    low_lim = window_center - window_length/2
    high_lim = window_center + window_length/2
    #conver to HU
    img = image * slope + intercept
    img[img < low_lim] = low_lim
    img[img > high_lim] = high_lim
    #map the image values to file format specifications
    img = (img - low_lim) * 255 / (high_lim - low_lim)
    img = img.astype(np.uint8)
    return img
    
mylist = [f for f in glob.glob("*.dcm")]

create_folder("input")
create_folder("target_full")
create_folder("target_noise")

for i in range(len(mylist)):
    #get dicom data
    ds = dicom.dcmread(mylist[i])
    #read image data
    pixel_array_numpy = ds.pixel_array
    #set file name of the file selected
    name = mylist[i][:-4]
    #create folder with the name of the file
    #create_folder(name)
    #change to that directory path
    #os.chdir(name)

    if "sub2" in name:
        os.chdir("input")
    elif "oise" in name:
        os.chdir("target_noise")
    else:
        os.chdir("target_full")

    for ii in range(pixel_array_numpy.shape[0]):
        img = to_image(pixel_array_numpy[ii,:,:])
        #cv2.imwrite(name + "_" + str(i) + ".bmp", img)
        np.save(name + "_" + str(ii) + ".npy", img)
    os.chdir("..")
    
    




