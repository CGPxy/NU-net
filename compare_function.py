#-*- conding:utf-8 -*-
import numpy as np
import tensorflow.keras.backend as K

import glob
import tqdm
from PIL import Image
import cv2
import os
from sklearn.metrics import f1_score
import tensorflow.compat.v1 as tf

import xlsxwriter
import surface_distance as surfdist

tf.disable_v2_behavior()

def get_contours(img):
    img_gray = cv2.cvtColor(img * 255, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    return contours[0]

def HD_base(y_true, y_pred):

    HD_cs1 = get_contours(y_true)
    HD_cs2 = get_contours(y_pred)

    # 3.创建计算距离对象
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    # 4.计算轮廓之间的距离
    d1 = hausdorff_sd.computeDistance(HD_cs1, HD_cs2)
    return d1

def ASSD(y_true, y_pred):
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)

    surface_distances = surfdist.compute_surface_distances(
        y_true, y_pred, spacing_mm=(1.0, 1.0, 1.0))
    avg_surf_dist = surfdist.compute_average_surface_distance(
        surface_distances)
    #(gt2pre, pre2gt)
    pre2gt = avg_surf_dist[1]
    return pre2gt

def get_boundary(img):
    showimg_gray = cv2.cvtColor(img * 255, cv2.COLOR_BGR2GRAY)
    showret, showbinary = cv2.threshold(showimg_gray, 50, 255, cv2.THRESH_BINARY)
    showcontours, showhierarchy = cv2.findContours(
        showbinary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mask_all = np.zeros((384, 384, 3))
    cv2.drawContours(mask_all, showcontours, -1, (1), 1)

    return mask_all


def ABD(y_true, y_pred):
    y_true = get_boundary(y_true)
    y_pred = get_boundary(y_pred)


    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)

    surface_distances = surfdist.compute_surface_distances(
        y_true, y_pred, spacing_mm=(1.0, 1.0, 1.0))
    avg_surf_dist = surfdist.compute_average_surface_distance(
        surface_distances)
    #(gt2pre, pre2gt)
    pre2gt = avg_surf_dist[1]
    return pre2gt

def cal_base(y_true, y_pred):
    y_pred_positive = K.round(K.clip(y_pred, 0, 1))
    y_pred_negative = 1 - y_pred_positive

    y_positive = K.round(K.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = K.sum(y_positive * y_pred_positive)
    TN = K.sum(y_negative * y_pred_negative)

    FP = K.sum(y_negative * y_pred_positive)
    FN = K.sum(y_positive * y_pred_negative)

    return TP, TN, FP, FN

def PA(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    ACC = (TP + TN) / (TP + FP + FN + TN + K.epsilon())
    return ACC


def IoU(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    iou = TP / (TP + FP + FN + K.epsilon())
    return iou


def Recall(y_true, y_pred):
    """ recall or sensitivity """
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SE = TP / (TP + FN + K.epsilon())
    return SE


def Precision(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    PC = TP / (TP + FP + K.epsilon())
    return PC


def Specificity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SP = TN / (TN + FP + K.epsilon())
    return SP


def F1_socre(y_true, y_pred):
    SE = Recall(y_true, y_pred)
    PC = Precision(y_true, y_pred)
    F1 = 2 * SE * PC / (SE + PC + K.epsilon())
    return F1

def Dice(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    DC = 2*TP/(2*TP+FP+FN)
    return DC

PAlist = []
MPAlist = []
IoUlist = []
MIoUlist = []
Precisionlist = []
Recalllist = []
Specificitylist = []
F1_socrelist = []
Dicelist = []
HDlist = []
ASSDlist = []
ABDlist = []


savepath = '/media/dy/Data_2T/CGP/Unet_Segnet/NU_net/result/excel/unet_MDS_MOU/BUSI/'
if not os.path.exists(savepath):
    os.makedirs(savepath)
excle1 = xlsxwriter.Workbook(savepath + "NUnet11.xlsx")
worksheet = excle1.add_worksheet()
worksheet.write(0,0,"image_name")
worksheet.write(0,1,"PA")
worksheet.write(0,2,"IoU")
worksheet.write(0,3,"Precision")
worksheet.write(0,4,"Recall")
worksheet.write(0,5,"Specificity")
worksheet.write(0,6,"F1_socre")
worksheet.write(0,7,"Dice")
worksheet.write(0,8,"HD")
worksheet.write(0,9,"ASSD")
worksheet.write(0,10,"ABD")

# predict mask
image_path = '/media/dy/Data_2T/CGP/Unet_Segnet/data/Breast/BUSI/new-BUSI/1/Test_images/labels/384/'

# Ground-turth mask
mask_path = '/media/dy/Data_2T/CGP/Unet_Segnet/NU_net/result/mask/unet_MDS_MOU/BUSI/1/'

filelist = os.listdir(mask_path)
i = 0
for item in filelist:
    print(item)
    i = i+1
    image1 = cv2.imread(image_path + item, cv2.IMREAD_GRAYSCALE)
    mask1 = cv2.imread(mask_path + item, cv2.IMREAD_GRAYSCALE)
    image=tf.cast(image1,tf.float32)
    mask=tf.cast(mask1,tf.float32)

    Himage = cv2.imread(image_path + item)
    Hmask = cv2.imread(mask_path + item)
    hd = HD_base(Himage, Hmask)
    assd = ASSD(Himage, Hmask)
    abd = ABD(Himage, Hmask)

    pa = PA(mask, image)
    iou = IoU(mask, image)
    precision = Precision(mask, image)
    recall = Recall(mask, image)
    specificity = Specificity(mask, image)
    f1_score = F1_socre(mask, image)
    dice = Dice(mask, image)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        PAlist.append(sess.run(pa))
        IoUlist.append(sess.run(iou))
        Precisionlist.append(sess.run(precision))
        Recalllist.append(sess.run(recall))
        Specificitylist.append(sess.run(specificity))
        F1_socrelist.append(sess.run(f1_score))
        Dicelist.append(sess.run(dice))
        HDlist.append(hd)
        ASSDlist.append(assd)
        ABDlist.append(abd)

        print(item)
        print("PA:%f   IoU:%f  Precision:%f  Recall:%f  Specificity:%f   F1_socre:%f    Dice:%f" 
            %(sess.run(pa), sess.run(iou), sess.run(precision), sess.run(recall), sess.run(specificity),
             sess.run(f1_score), sess.run(dice)))

        worksheet.write(i,0,item)
        worksheet.write(i,1,sess.run(pa))
        worksheet.write(i,2,sess.run(iou))
        worksheet.write(i,3,sess.run(precision))
        worksheet.write(i,4,sess.run(recall))
        worksheet.write(i,5,sess.run(specificity))
        worksheet.write(i,6,sess.run(f1_score))
        worksheet.write(i,7,sess.run(dice))
        worksheet.write(i,8,hd)
        worksheet.write(i,9,assd)
        worksheet.write(i,10,abd)

if i != 0:

    worksheet.write(i+2,1,sum(PAlist) / i)
    worksheet.write(i+2,2,sum(IoUlist) / i)
    worksheet.write(i+2,3,sum(Precisionlist) / i)
    worksheet.write(i+2,4,sum(Recalllist) / i)
    worksheet.write(i+2,5,sum(Specificitylist) / i)
    worksheet.write(i+2,6,sum(F1_socrelist) / i)
    worksheet.write(i+2,7,sum(Dicelist) / i)
    worksheet.write(i+2,8,sum(HDlist) / i)
    worksheet.write(i+2,9,sum(ASSDlist) / i)
    worksheet.write(i+2,10,sum(ABDlist) / i)   
    excle1.close()

    print("MPA:%f" % (sum(PAlist) / i))
    print("MIoU:%f" % (sum(IoUlist) / i))
    print("MPrecision:%f" % (sum(Precisionlist) / i))
    print("MRecall:%f" % (sum(Recalllist) / i))
    print("MSpecificity:%f" % (sum(Specificitylist) / i))
    print("MF1_score:%f" % (sum(F1_socrelist) / i))
    print("Dice:%f" % (sum(Dicelist) / i))
    print("HD:%f" % (sum(HDlist) / i))
    print("ASSD:%f" % (sum(ASSDlist) / i))
    print("ABD:%f" % (sum(ABDlist) / i))
