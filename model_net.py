# coding=utf-8
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import cv2
import random
import os
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *   
from tensorflow.keras.layers import *   
from tensorflow.python.layers import utils
from tensorflow.keras import regularizers



img_w = 384  
img_h = 384


def ConvBlock(data, filte):
    conv1 = Conv2D(filte, (3, 3), padding="same")(data) #,dilation_rate=(4,4)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)
    conv2 = Conv2D(filte, (3, 3), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = LeakyReLU(alpha=0.01)(batch2)
    return LeakyReLU2



def updata1(data, skipdata, filte):
    shape = K.int_shape(skipdata)
    shape1 = K.int_shape(data)

    data1 = UpSampling2D((shape[1] // shape1[1], shape[2] // shape1[2]))(data)
    concatenate = Concatenate()([skipdata, data1])
    concatenate = ConvBlock(data=concatenate, filte=filte)

    return concatenate


def deep_unet7():
    inputs = Input((img_h, img_w, 3))

    Conv1 = ConvBlock(data=inputs, filte=64)

    pool1 = MaxPooling2D(pool_size=(2, 2))(Conv1)
    Conv2 = ConvBlock(data=pool1, filte=128)

    pool21 = MaxPooling2D(pool_size=(4, 4))(Conv1)    
    Conv31 = ConvBlock(data=pool21, filte=64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(Conv2)
    Conv3 = Conv2D(32, (3, 3), padding="same")(pool21)
    Conv3 = BatchNormalization()(Conv3)
    Conv3 = LeakyReLU(alpha=0.01)(Conv3)
    Conv3 = Concatenate()([Conv3, pool2])
    Conv3 = ConvBlock(data=Conv3, filte=128)


    pool3 = MaxPooling2D(pool_size=(2, 2))(Conv3)   
    Conv4 = ConvBlock(data=pool3, filte=256)
    pool31 = MaxPooling2D(pool_size=(4, 4))(Conv2)    
    Conv41 = ConvBlock(data=pool31, filte=128)
    pool32 = MaxPooling2D(pool_size=(2, 2))(Conv31)    
    Conv42 = ConvBlock(data=pool32, filte=64)


    pool41 = MaxPooling2D(pool_size=(4, 4))(Conv3)    
    Conv51 = ConvBlock(data=pool41, filte=128)
    pool42 = MaxPooling2D(pool_size=(2, 2))(Conv41)    
    Conv52 = ConvBlock(data=pool42, filte=128)
    pool43 = MaxPooling2D(pool_size=(2, 2))(Conv42)    
    Conv53 = ConvBlock(data=pool43, filte=64)
    pool4 = MaxPooling2D(pool_size=(2, 2))(Conv4) 
    Conv5 = Conv2D(32, (3, 3), padding="same")(pool41)
    Conv5 = BatchNormalization()(Conv5)
    Conv5 = LeakyReLU(alpha=0.01)(Conv5)
    Conv5 = Concatenate()([Conv5, pool4]) 
    Conv5 = ConvBlock(data=Conv5, filte=256)


    pool5 = MaxPooling2D(pool_size=(2, 2))(Conv5)    
    Conv6 = ConvBlock(data=pool5, filte=512)
    pool51 = MaxPooling2D(pool_size=(4, 4))(Conv4)    
    Conv61 = ConvBlock(data=pool51, filte=256)
    pool52 = MaxPooling2D(pool_size=(2, 2))(Conv51)    
    Conv62 = ConvBlock(data=pool52, filte=128)
    pool53 = MaxPooling2D(pool_size=(2, 2))(Conv52)    
    Conv63 = ConvBlock(data=pool53, filte=128)
    pool54 = MaxPooling2D(pool_size=(2, 2))(Conv53)    
    Conv64 = ConvBlock(data=pool54, filte=64)


    pool61 = MaxPooling2D(pool_size=(4, 4))(Conv5)    
    Conv71 = ConvBlock(data=pool61, filte=256)
    pool62 = MaxPooling2D(pool_size=(2, 2))(Conv61)    
    Conv72 = ConvBlock(data=pool62, filte=256)
    pool63 = MaxPooling2D(pool_size=(2, 2))(Conv62)    
    Conv73 = ConvBlock(data=pool63, filte=128)
    pool64 = MaxPooling2D(pool_size=(2, 2))(Conv63)    
    Conv74 = ConvBlock(data=pool64, filte=128)
    pool65 = MaxPooling2D(pool_size=(2, 2))(Conv64)    
    Conv75 = ConvBlock(data=pool65, filte=64)
    pool6 = MaxPooling2D(pool_size=(2, 2))(Conv6)
    Conv7 = Conv2D(32, (3, 3), padding="same")(pool61)
    Conv7 = BatchNormalization()(Conv7)
    Conv7 = LeakyReLU(alpha=0.01)(Conv7)
    Conv7 = Concatenate()([Conv7, pool6]) 
    Conv7 = ConvBlock(data=Conv7, filte=512)


    pool7 = MaxPooling2D(pool_size=(2, 2))(Conv7)    
    Conv8 = ConvBlock(data=pool7, filte=1024)
    pool71 = MaxPooling2D(pool_size=(4, 4))(Conv6)    
    Conv81 = ConvBlock(data=pool71, filte=512)
    pool72 = MaxPooling2D(pool_size=(2, 2))(Conv71)    
    Conv82 = ConvBlock(data=pool72, filte=256)
    pool73 = MaxPooling2D(pool_size=(2, 2))(Conv72)    
    Conv83 = ConvBlock(data=pool73, filte=256)
    # pool74 = MaxPooling2D(pool_size=(4, 4))(Conv61)    
    # Conv84 = ConvBlock(data=pool74, filte=256)
    pool74 = MaxPooling2D(pool_size=(2, 2))(Conv73)    
    Conv84 = ConvBlock(data=pool74, filte=128)
    pool75 = MaxPooling2D(pool_size=(2, 2))(Conv74)    
    Conv85 = ConvBlock(data=pool75, filte=128)
    pool76 = MaxPooling2D(pool_size=(2, 2))(Conv75)    
    Conv86 = ConvBlock(data=pool76, filte=64)


    # 6
    concatenate1 = Concatenate()([Conv8, Conv81, Conv82, Conv83, Conv84, Conv85, Conv86])
    up1 = updata1(filte=512, data=concatenate1, skipdata=Conv7)

    # 12
    updata01 = UpSampling2D((4, 4))(Conv81)
    concatenate2 = Concatenate()([Conv6, updata01])
    up2 = updata1(filte=512, data=up1, skipdata=concatenate2)

    # 24
    updata2 = UpSampling2D((2, 2))(Conv82)
    updata2 = ConvBlock(data=updata2, filte=256)
    updata3 = UpSampling2D((4, 4))(updata2)
    concatenate3 = Concatenate()([Conv5, updata3])
    up3 = updata1(filte=256, data=up2, skipdata=concatenate3)

    # 48
    updata4 = UpSampling2D((2, 2))(Conv83)
    updata4 = ConvBlock(data=updata4, filte=256)
    updata4 = UpSampling2D((2, 2))(updata4)
    updata4 = ConvBlock(data=updata4, filte=256)
    updata5 = UpSampling2D((4, 4))(updata4)
    concatenate4 = Concatenate()([Conv4, updata5])
    up4 = updata1(filte=256, data=up3, skipdata=concatenate4)

    # 96
    updata6 = UpSampling2D((2, 2))(Conv84)
    updata6 = ConvBlock(data=updata6, filte=128)
    updata6 = UpSampling2D((2, 2))(updata6)
    updata6 = ConvBlock(data=updata6, filte=128)
    updata6 = UpSampling2D((2, 2))(updata6)
    updata6 = ConvBlock(data=updata6, filte=128)
    updata7 = UpSampling2D((4, 4))(updata6)
    concatenate5 = Concatenate()([Conv3, updata7])
    up5 = updata1(filte=128, data=up4, skipdata=concatenate5)

    # 192
    updata8 = UpSampling2D((2, 2))(Conv85)
    updata8 = ConvBlock(data=updata8, filte=128)
    updata8 = UpSampling2D((2, 2))(updata8)
    updata8 = ConvBlock(data=updata8, filte=128)
    updata8 = UpSampling2D((2, 2))(updata8)
    updata8 = ConvBlock(data=updata8, filte=128)
    updata8 = UpSampling2D((2, 2))(updata8)
    updata8 = ConvBlock(data=updata8, filte=128)
    updata9 = UpSampling2D((4, 4))(updata8)
    concatenate6 = Concatenate()([Conv2, updata9])
    up6 = updata1(filte=128, data=up5, skipdata=concatenate6)

    # 384
    updata10 = UpSampling2D((2, 2))(Conv86)
    updata10 = ConvBlock(data=updata10, filte=64)
    updata10 = UpSampling2D((2, 2))(updata10)
    updata10 = ConvBlock(data=updata10, filte=64)
    updata10 = UpSampling2D((2, 2))(updata10)
    updata10 = ConvBlock(data=updata10, filte=64)
    updata10 = UpSampling2D((2, 2))(updata10)
    updata10 = ConvBlock(data=updata10, filte=64)
    updata10 = UpSampling2D((2, 2))(updata10)
    updata10 = ConvBlock(data=updata10, filte=64)
    updata11 = UpSampling2D((4, 4))(updata10)
    concatenate7 = Concatenate()([Conv1, updata11])
    up7 = updata1(filte=64, data=up6, skipdata=concatenate7)

    outconv = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(up7)
    out = Activation('sigmoid')(outconv)

    model = Model(inputs=inputs, outputs=out)
    return model