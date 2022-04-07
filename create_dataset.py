import glob
import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt
import os
from util import create_folder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="", help="choose directory to save the dataset")
opt = parser.parse_args()

def to_image(image, slope=0.0641, intercept=-1200, window_length=100, window_center=40):
    #Images have on average mean of 660.69 and std of 630.22
    #conver to HU
    img = image * slope + intercept
    # Scale the images from 0 to 1
    img = (img + 1200) / (1200 + 3000.793501)
    # Substract the average and divide by the std() of the average over all imags
    img = (img - 0.12838) / 0.15002428 
    #The code below is for windowing, not needed in this case
    #calculate limitis to clip the image
    #low_lim = window_center - window_length/2
    #high_lim = window_center + window_length/2
    #img = image * slope + intercept
    #img[img < low_lim] = low_lim
    #img[img > high_lim] = high_lim
    #img = (img - low_lim) * 255 / (high_lim - low_lim)
    #img = img.astype(np.uint8)
    return img
    
mylist = [f for f in glob.glob("*.dcm")]

create_folder("input", opt.dir)
create_folder("target_full", opt.dir)
create_folder("target_noise", opt.dir)

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
    
    




