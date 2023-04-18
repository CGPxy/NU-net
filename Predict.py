import cv2
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import argparse
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
import tensorflow as tf
# from keras.layers.core import Lambda
from our_train import *
mpl.use('TkAgg')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEST_SET = []

def read_directory(args):
    for filename in os.listdir(args["Imagepath"]):
        # image
#        image = cv2.imread(directory_name + "/" + filename)
        TEST_SET.append(filename)

def BCE():
    def dice(y_true, y_pred):
        return tf.keras.metrics.binary_crossentropy(y_true, y_pred)
    return dice

def original_predict(args):
    # load the trained convolutional neural network
    model = load_model(args["model"], custom_objects={'dice':BCE()})
    model.summary()

    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        # BUSI
        image = cv2.imread(args["Imagepath"] + path)
        image = np.array(image,dtype=np.uint8)
        # image = img_to_array(image)
        h,w,_ = image.shape
        
        padding_h = h
        padding_w = w
        padding_img = np.zeros((padding_h, padding_w, 3),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)

        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
   
        crop = padding_img[:,:,:3]
            
        crop = np.expand_dims(crop, axis=0) 
        pred = model.predict(crop,verbose=2)

        preimage = pred.reshape((384,384))

        h,w = preimage.shape
        for i in range(0, h):
            for j in range(0, w):
                if (preimage[i, j] > 0.5):
                    preimage[i, j] = 1
                else:
                    preimage[i, j] = 0

        pred = preimage.reshape((384,384)).astype(np.uint8)
        mask_whole[:,:] = pred[:,:]

        savepath = args["Saveimagepath"]
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        cv2.imwrite(savepath+path,mask_whole[0:h,0:w])

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to trained model model")
    ap.add_argument("--Imagepath", required=True, help="path of your test image")
    ap.add_argument("--Saveimagepath", required=True, help="savepath of your predict mask")
    args = vars(ap.parse_args())    
    return args

if __name__ == '__main__':
    args = args_parse()
    original_predict(args)
    read_directory(args)
