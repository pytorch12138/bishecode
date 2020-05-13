# import os
# import cv2
# import numpy.random as ra
# img_path_list=[]
# label_path_list=[]
# def get_full_path(img_list,label_list):
#    img_path_list=[]
#    label_path_list=[]
#    for img_index,label_index in zip(img_list,label_list):
#        img_path_list.append('./house/test/house/imgs/'+img_index)
#        label_path_list.append('./house/test/house/labels/'+label_index)
#    return img_path_list,label_path_list
# img_list=os.listdir('./house/test/house/imgs/')
# label_list=os.listdir('./house/test/house/labels/')
# img_list.sort(key=lambda  x:int(x[:-4]))
# label_list.sort(key=lambda  x:int(x[:-4]))
# print(img_list)
# # pos_n_size=200*len(img_list)
# # neg_n_size=200*len(img_list)
# s_size=16
# _0_index=0
# _1_index=0
# _2_index=500
# _3_index=12000
# _4_index=12000
# pos_n_size=200
# img_path_list,label_path_list=get_full_path(img_list,label_list)
# a1_index=0
# a2_index=0
# for img_index,label_index in zip(img_path_list,label_path_list):
#     img=cv2.imread(img_index)
#     label=cv2.imread(label_index,0)
#     print(img_index)
#     print(label_index)
#     img_width,img_hegiht=label.shape
#     _0_index=0
#     _1_index=0
#     while   _0_index<pos_n_size or  _1_index<pos_n_size:
#         x_pos=ra.randint(0,img_width-s_size,1)[0]
#         y_pos=ra.randint(0,img_hegiht-s_size,1)[0]
#         label_roi=label[x_pos:(x_pos+s_size),y_pos:(y_pos+s_size)]
#         img_roi=img[x_pos:x_pos+s_size,y_pos:y_pos+s_size,:]
#         print((label_roi==1).all())
#         if _0_index <= pos_n_size:
#              if (label_roi == 1).all():
#                  cv2.imwrite('./data/imgs/{}.png'.format(a1_index),img_roi)
#                  _0_index+=1
#                  a1_index+=1
#         if _1_index<=pos_n_size:
#                 if (label_roi == 0).all():
#                     cv2.imwrite('./data/others/{}.png'.format(a2_index),img_roi)
#                     _1_index+=1
#                     a2_index+=1

#
# #
# #     if _1_index < pos_n_size:
# #         if (label_roi == 1).all():
# #             cv2.imwrite('./svm_data/1/{}.png'.format(_1_index), img_roi)
# #             _1_index += 1
# #     if _2_index < pos_n_size:
# #         if (label_roi == 2).all():
# #             cv2.imwrite('./svm_data/2/{}.png'.format(_2_index), img_roi)
# #             _2_index += 1
# #     if _3_index < pos_n_size:
# #         if (label_roi == 3).all():
# #             cv2.imwrite('./svm_data/3/{}.png'.format(_3_index), img_roi)
# #             _3_index += 1
# #     if _4_index < pos_n_size:
# #         if (label_roi == 4).all():
# #             cv2.imwrite('./svm_data/4/{}.png'.format(_4_index), img_roi)
# #             _4_index += 1
# #     if _0_index>=pos_n_size and _1_index>=pos_n_size and _2_index>=pos_n_size and _3_index>=pos_n_size and _4_index>=pos_n_size:
# #         break
# # #         if pos_index<pos_n_size:
# # #             if (label_roi==1).all():
# # #                 print("生成一张正样本")
# # #                 cv2.imwrite('./data/house/imgs/{}.png'.format(pos_index),img_roi)
# # #                 pos_index+=1
# # #                 print(pos_index)
# # #         if neg_index<neg_n_size:
# # #             if (label_roi==0).all():
# # #                 print("生成一张负样本")
# # #                 cv2.imwrite('./data/house/other/{}.png'.format(neg_index),img_roi)
# # #                 neg_index+=1
# # #                 print(neg_index)
# #
# import cv2
# from sklearn.externals import joblib
# from sklearn.svm import SVC
# import math
# from pywt import dwt2,wavedec2
# import cv2
# import numpy as np
# import os
# #定义最大灰度级数
# gray_level=8
# def maxGray_level(img):
#     max_gray_level=0
#     height,width=img.shape
#     for y in range(height):
#         for x in range(width):
#             if img[y][x]>max_gray_level:
#                 max_gray_level = img[y][x]
#     return max_gray_level + 1
#
#
# def getGlcm(input, d_x, d_y):
#     srcdata = input.copy()
#     ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
#     (height, width) = input.shape
#     max_gray_level = maxGray_level(input)
# # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
#     if max_gray_level > gray_level:
#         for j in range(height):
#             for i in range(width):
#                 srcdata[j][i] = srcdata[j][i]*gray_level / max_gray_level
#     if (d_x >=0 and d_y>=0):
#      for j in range(height-d_y):
#         for i in range(width-d_x):
#             rows = srcdata[j][i]
#             cols = srcdata[j + d_y][i + d_x]
#             ret[rows][cols] += 1.0
#     elif(d_x <0 and d_y>0):
#         for j in range(height - d_y):
#             for i in range(width +d_x):
#                 rows = srcdata[j][width-i-1]
#                 cols = srcdata[j + d_y][width-i+d_x]
#                 ret[rows][cols] += 1.0
#     elif (d_x > 0 and d_y < 0):
#         for j in range(height +d_y):
#             for i in range(width-d_x):
#                 rows = srcdata[height-j-1][i]
#                 cols = srcdata[height-j + d_y][i + d_x]
#                 ret[rows][cols] += 1.0
#     elif (d_x < 0 and d_y < 0):
#         for j in range(height +d_y):
#             for i in range(width+d_x):
#                 rows = srcdata[height-j-1][width-i-1]
#                 cols = srcdata[height-j + d_y][width-i + d_x]
#                 ret[rows][cols] += 1.0
#     for i in range(gray_level):
#         for j in range(gray_level):
#             ret[i][j] /= float(height * width)
#     return ret
# def feature_computer(p):
#     Con = 0.0
#     Eng = 0.0
#     Asm = 0.0
#     Idm = 0.0
#     for i in range(gray_level):
#         for j in range(gray_level):
#             Con += (i - j) * (i - j) * p[i][j]
#             Asm += p[i][j] * p[i][j]
#             Idm += p[i][j] / (1 + (i - j) * (i - j))
#             if p[i][j] > 0.0:
#                 Eng += p[i][j] * math.log(p[i][j])
#     return Asm, Con, -Eng, Idm
# def get_img_features(img):
#     (B, G, R) = cv2.split(img)
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     glcm_0 = getGlcm(img, 1, 0)
#     glcm_1 = getGlcm(img, 0, 1)
#     glcm_3 = getGlcm(img, 1, 1)
#     glcm_4 = getGlcm(img, -1,-1)
#     asm,con,eng,idm=feature_computer(glcm_0)
#     asm1, con1, eng1, idm1 = feature_computer(glcm_1)
#     asm2, con2, eng2, idm2 = feature_computer(glcm_3)
#     asm3, con3, eng3, idm3 = feature_computer(glcm_4)
#     img1=img.astype(np.float32)
#     coeffs = wavedec2(img1, 'haar', level=2)
#     cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
#     m_ca=np.mean(cA2)
#     std_ch=np.std(cH2) + np.std(cH1)
#     std_cv=np.std(cV2) + np.std(cV1)
#     std_cd=np.std(cD2) + np.std(cD1)
#     return [(asm+asm1+asm2+asm3)/4,(con+con1+con2+con3)/4,(eng+eng1+eng2+eng3)/4,(idm+idm1+idm2+idm3)/4,m_ca,std_ch,np.mean(R),np.std(R),np.mean(G),np.std(G),np.mean(B),np.std(B)]
#
# img_path_list=[]
# label_path_list=[]
# def get_full_path(img_list,img1_list):
#    img_path_list=[]
#    img1_path_list = []
#    img2_path_list = []
#    img3_path_list = []
#    img4_path_list = []
#    for img_index,img1_index in zip(img_list,img1_list):
#        img_path_list.append('./data/imgs/'+img_index)
#        img1_path_list.append('./data/others/'+img1_index)
#        # img2_path_list.append('./svm_data/2/' + img2_index)
#        # img3_path_list.append('./svm_data/3/' + img3_index)
#        # img4_path_list.append('./svm_data/4/' + img4_index)
#    return img_path_list,img1_path_list
#        #
#        # ,img2_path_list,img3_path_list,img4_path_list
# img_list=os.listdir('./data/imgs/')
# img1_list=os.listdir('./data/others/')
# # # img2_list=os.listdir('./svm_data/2/')
# # # img3_list=os.listdir('./svm_data/3/')
# # # img4_list=os.listdir('./svm_data/4/')
# img_list.sort(key=lambda  x:int(x[:-4]))
# img1_list.sort(key=lambda  x:int(x[:-4]))
# # # img2_list.sort(key=lambda  x:int(x[:-4]))
# # # img3_list.sort(key=lambda  x:int(x[:-4]))
# # # img4_list.sort(key=lambda  x:int(x[:-4]))
# img_path_list,img1_path_list=get_full_path(img_list,img1_list)
# svm_label_list=[]
# feature_list=[]
# for img_index,img1_index in zip(img_path_list,img1_path_list):
#     print(img1_index)
#     print(img_index)
#     img = cv2.imread(img_index)
#     img1=cv2.imread(img1_index)
# #     # img2 = cv2.imread(img2_index)
# #     # img3 = cv2.imread(img3_index)
# #     # img4 = cv2.imread(img4_index)
#     img0_feature_list=get_img_features(img)
#     svm_label_list.append(1)
#     with open('./h_label.txt','a',encoding='utf-8') as f:
#             f.writelines(str(img0_feature_list[0])+" "+str(img0_feature_list[1])+" "+str(img0_feature_list[2])+" "+str(img0_feature_list[3])+" "+str(img0_feature_list[4])+" "+str(img0_feature_list[5])+" "+str(img0_feature_list[6])+" "+str(img0_feature_list[7])+" "+str(img0_feature_list[8])+" "+str(img0_feature_list[9])+" "+str(img0_feature_list[10])+" "+str(img0_feature_list[11])+" "+str(svm_label_list[0])+'\n')
#     img1_feature_list = get_img_features(img1)
#     svm_label_list.append(-1)
#     with open('./h_label.txt', 'a', encoding='utf-8') as f:
#         f.writelines(
#             str(img1_feature_list[0]) + " " + str(img1_feature_list[1]) + " " + str(img1_feature_list[2]) + " " + str(
#                 img1_feature_list[3]) + " " + str(img1_feature_list[4]) + " " + str(img1_feature_list[5]) + " " + str(
#                 img1_feature_list[6]) + " " + str(img1_feature_list[7]) + " "+str(img1_feature_list[8])+" "+str(img1_feature_list[9])+" "+str(img1_feature_list[10])+" "+str(img1_feature_list[11])+" " + str(svm_label_list[1]) + '\n')
#         svm_label_list.clear()
#     # img2_feature_list = get_img_features(img2)
#     # svm_label_list.append(2)
#     # with open('./c_label.txt', 'a', encoding='utf-8') as f:
#     #     f.writelines(
#     #         str(img2_feature_list[0]) + " " + str(img2_feature_list[1]) + " " + str(img2_feature_list[2]) + " " + str(
#     #             img2_feature_list[3]) + " " + str(img2_feature_list[4]) + " " + str(img2_feature_list[5]) + " " + str(
#     #             img2_feature_list[6]) + " " + str(img2_feature_list[7]) + " " +str(img2_feature_list[8])+" "+str(img2_feature_list[9])+" "+str(img2_feature_list[10])+" "+str(img2_feature_list[11])+" "+str(svm_label_list[2]) + '\n')
#     # img3_feature_list = get_img_features(img3)
#     # svm_label_list.append(3)
#     # with open('./c_label.txt', 'a', encoding='utf-8') as f:
#     #     f.writelines(
#     #         str(img3_feature_list[0]) + " " + str(img3_feature_list[1]) + " " + str(img3_feature_list[2]) + " " + str(
#     #             img3_feature_list[3]) + " " + str(img3_feature_list[4]) + " " + str(img3_feature_list[5]) + " " + str(
#     #             img3_feature_list[6]) + " " + str(img3_feature_list[7]) + " " + str(img3_feature_list[8])+" "+str(img3_feature_list[9])+" "+str(img3_feature_list[10])+" "+str(img3_feature_list[11])+" "+str(svm_label_list[3]) + '\n')
#     # img4_feature_list = get_img_features(img4)
#     # svm_label_list.append(4)
#     # with open('./c_label.txt', 'a', encoding='utf-8') as f:
#     #     f.writelines(
#     #         str(img4_feature_list[0]) + " " + str(img4_feature_list[1]) + " " + str(img4_feature_list[2]) + " " + str(
#     #             img4_feature_list[3]) + " " + str(img4_feature_list[4]) + " " + str(img4_feature_list[5]) + " " + str(img4_feature_list[6]) + " " + str(img4_feature_list[7]) + " " +str(img4_feature_list[8])+" "+str(img4_feature_list[9])+" "+str(img4_feature_list[10])+" "+str(img4_feature_list[11])+" "+ str(svm_label_list[4]) + '\n')
#     # svm_label_list.clear()
#
# #
# #
#     river_feature_list = get_img_features(img)
#     svm_label_list.append(1)
#     with open('./h_label.txt','a',encoding='utf-8') as f:
#         f.writelines(str(river_feature_list[0])+" "+str(river_feature_list[1])+" "+str(river_feature_list[2])+" "+str(river_feature_list[3])+" "+str(river_feature_list[4])+" "+str(river_feature_list[5])+" "+str(river_feature_list[6])+" "+str(river_feature_list[7])+" "+str(river_feature_list[8])+" "+str(river_feature_list[9])+" "+str(river_feature_list[10])+" "+str(river_feature_list[11])+" "+str(svm_label_list[0])+'\n')
#     other_feature_list = get_img_features(img1)
#     svm_label_list.append(-1)
#     with open('./h_label.txt','a',encoding='utf-8') as f:
#         f.writelines(str(other_feature_list[0])+" "+str(other_feature_list[1])+" "+str(other_feature_list[2])+" "+str(other_feature_list[3])+" "+str(other_feature_list[4])+" "+str(other_feature_list[5])+" "+str(other_feature_list[6])+" "+str(other_feature_list[7])+" "+str(other_feature_list[8])+" "+str(other_feature_list[9])+" "+str(other_feature_list[10])+" "+str(other_feature_list[11])+" "+str(svm_label_list[1])+'\n')
#     svm_label_list.clear()
# # #
# # # feature_list.append(other_feature_list)
# # # feature_array=np.array(feature_list)
# # # svm_label_array=np.array(svm_label_list)
# # # for feature_index,label_index in zip(feature_list,svm_label_array):
# # #  pass
# #
# # # svclassifier = SVC(kernel='poly', degree=8)
# # # svclassifier.fit(feature_array, svm_label_array)
# # # joblib.dump(svclassifier,'./svm.pkl')
# #
# # import os
# # from sklearn.externals import joblib
# # from sklearn.svm import SVC
# # from sklearn.model_selection import GridSearchCV
# # import numpy as np
# # features_list=[]
# # label_list=[]
# # with open('./label.txt','r',encoding='utf-8') as f:
# #     feature_and_label_=f.readlines()
# # for aa in feature_and_label_:
# #     label_list.append(float(aa.strip('\n').split(' ',)[-1]))
# #     features_list.append([float(f) for f in aa.strip('\n').split(' ',)[0:6]])
# # features_array=np.array(features_list)
# # label_array=np.array(label_list)
# # print("开始")
# # svclassifier = SVC(kernel='poly')
# # parameters = {'C':[0.5,1, 2, 4], 'degree':[1,2,4,8]}
# # clf = GridSearchCV(svclassifier, parameters, scoring='f1')
# # svclassifier.fit(features_array, label_array)
# # print('The parameters of the best model are: ')
# # print(clf.best_params_)
# # joblib.dump(svclassifier,'./svm.pkl')
# # test_features=features_array[0:20,:]
# # test_label=label_array[0:20]
# # pred_=svclassifier.predict(test_features)
# # print("predict")
# # print(pred_)
# # print("label")
# # print(test_label)
#
#
#
#
#
#
#
#
#
#
#
#
# from sklearn.externals import joblib
# from sklearn.svm import SVC
# import matplotlib.pylab as plt
# import cv2
# import os
# import numpy as np
# img_path_list=[]
# label_path_list=[]
# def get_full_path(img_list,label_list):
#    img_path_list=[]
#    label_path_list=[]
#    for img_index,label_index in zip(img_list,label_list):
#        img_path_list.append('./river/imgs/'+img_index)
#        label_path_list.append('./river/labels/'+label_index)
#    return img_path_list,label_path_list
# img_list=os.listdir('./river/imgs/')
# label_list=os.listdir('./river/labels/')
# img_list.sort(key=lambda  x:int(x[:-4]))
# label_list.sort(key=lambda  x:int(x[:-4]))
# img_path_list,label_path_list=get_full_path(img_list,label_list)
# imgs_list=[]
# labels_list=[]
# for img_index,label_index in zip(img_path_list,label_path_list):
#     img=cv2.imread(img_index)
#     img = img.reshape(-1,3)
#     label=cv2.imread(label_index,0)
#     label=2*(label/255)-1
#     label=label.reshape(-1)
#     imgs_list.append(img)
#     labels_list.append(label)
# imgs=np.concatenate(imgs_list[:3],axis=0)
# labels=np.concatenate(labels_list[:3],axis=0)
# print(imgs.shape)
# svclassifier = SVC(kernel='poly', degree=8)
# svclassifier.fit(imgs, labels,)
# img = cv2.imread('./river/imgs/0.png')
# img = img.reshape(-1, 3)
# y_pred=svclassifier.predict(img)
# print(y_pred.shape)
# y_pred=y_pred.reshape(256,256)
# y_pred(0.5*y_pred+0.5)*255
# y_pred=np.array(y_pred,dtype=np.uint8)
# cv2.imwrite('./0.png',y_pred)
# joblib.dump(svclassifier,'./svm.pkl')
#
#
#
#
#
#
#
# # X, y = make_blobs(n_samples=100, centers=3,
# #                   random_state=0, cluster_std=0.8)
# # print(X,y)
# # clf_linear = svm.SVC(C=1.0, kernel='linear')
# # clf_poly = svm.SVC(C=1.0, kernel='poly', degree=3)
# # clf_rbf = svm.SVC(C=1.0, kernel='rbf', gamma=0.5)
# # clf_rbf2 = svm.SVC(C=1.0, kernel='rbf', gamma=0.1)
# #
# # plt.figure(figsize=(10, 10), dpi=144)
# #
# # clfs = [clf_linear, clf_poly, clf_rbf, clf_rbf2]
# # titles = ['Linear Kernel',
# #           'Polynomial Kernel with Degree=3',
# #           'Gaussian Kernel with $\gamma=0.5$',
# #           'Gaussian Kernel with $\gamma=0.1$']
# # for clf, i in zip(clfs, range(len(clfs))):
# #     clf.fit(X, y)
# # #     plt.subplot(2, 2, i+1)
#
import cv2
from sklearn.externals import joblib
from sklearn.svm import SVC
import math
from pywt import dwt2,wavedec2
import cv2
import numpy as np
import os
from evaluate import evlate
#定义最大灰度级数
gray_level=8
def maxGray_level(img):
    max_gray_level=0
    height,width=img.shape
    for y in range(height):
        for x in range(width):
            if img[y][x]>max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1


def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape
    max_gray_level = maxGray_level(input)
# 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i]*gray_level / max_gray_level
    if (d_x >=0 and d_y>=0):
     for j in range(height-d_y):
        for i in range(width-d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0
    elif(d_x <0 and d_y>0):
        for j in range(height - d_y):
            for i in range(width +d_x):
                rows = srcdata[j][width-i-1]
                cols = srcdata[j + d_y][width-i+d_x]
                ret[rows][cols] += 1.0
    elif (d_x > 0 and d_y < 0):
        for j in range(height +d_y):
            for i in range(width-d_x):
                rows = srcdata[height-j-1][i]
                cols = srcdata[height-j + d_y][i + d_x]
                ret[rows][cols] += 1.0
    elif (d_x < 0 and d_y < 0):
        for j in range(height +d_y):
            for i in range(width+d_x):
                rows = srcdata[height-j-1][width-i-1]
                cols = srcdata[height-j + d_y][width-i + d_x]
                ret[rows][cols] += 1.0
    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)
    return ret
def feature_computer(p):
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Eng, Idm
def get_img_features(img):
    (B, G, R) = cv2.split(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    glcm_0 = getGlcm(img, 1, 0)
    glcm_1 = getGlcm(img, 0, 1)
    glcm_3 = getGlcm(img, 1, 1)
    glcm_4 = getGlcm(img, -1,-1)
    asm,con,eng,idm=feature_computer(glcm_0)
    asm1, con1, eng1, idm1 = feature_computer(glcm_1)
    asm2, con2, eng2, idm2 = feature_computer(glcm_3)
    asm3, con3, eng3, idm3 = feature_computer(glcm_4)
    img1=img.astype(np.float32)
    coeffs = wavedec2(img1, 'haar', level=2)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    m_ca=np.mean(cA2)
    std_ch=np.std(cH2) + np.std(cH1)
    std_cv=np.std(cV2) + np.std(cV1)
    std_cd=np.std(cD2) + np.std(cD1)
    return [(asm+asm1+asm2+asm3)/4,(con+con1+con2+con3)/4,(eng+eng1+eng2+eng3)/4,(idm+idm1+idm2+idm3)/4,m_ca,std_ch,np.mean(R),np.std(R),np.mean(G),np.std(G),np.mean(B)]

import cv2
import numpy as np
from sklearn.externals import joblib
from evaluate import evlate
s_size=16
clf=joblib.load('./svm_model/svm_color.pkl')
img_list=os.listdir('./test/imgs/')
img_list.sort(key=lambda x: int(x[:-4]))
print(img_list)
imgs_list=[]
labels_list=[]
for img_folder in img_list:
 imgs_list.append('./test/imgs/'+img_folder)
 labels_list.append('./test/label/'+img_folder)
for (img_index,labels_index) in zip(imgs_list,labels_list):
 img = cv2.imread(img_index)
 w, h, _ = img.shape
 print(w, h)
 label = np.zeros(shape=(w, h), dtype=np.uint8)
 for w_index in range(int(int(w-s_size)/int(s_size))+1):
    for h_index in range(int(int(h-s_size)/int(s_size))+1):
        img_roi=img[int(w_index*s_size):int(s_size+w_index*s_size),int(h_index*s_size):int(s_size+h_index*s_size),:]
        feature_list=get_img_features(img_roi)
        feature_array=np.array([feature_list])
        pred=clf.predict(feature_array)
        if pred[0]==0:
         label[int(w_index*s_size):int(s_size+w_index*s_size),int(h_index*s_size):int(s_size+h_index*s_size)]=0
        if pred[0]==1:
         label[int(w_index*s_size):int(s_size+w_index*s_size),int(h_index*s_size):int(s_size+h_index*s_size)]=1
        if pred[0]==2:
         label[int(w_index*s_size):int(s_size+w_index*s_size),int(h_index*s_size):int(s_size+h_index*s_size)]=2
        if pred[0]==3:
         label[int(w_index*s_size):int(s_size+w_index*s_size),int(h_index*s_size):int(s_size+h_index*s_size)]=3
        if pred[0]==4:
         label[int(w_index*s_size):int(s_size+w_index*s_size),int(h_index*s_size):int(s_size+h_index*s_size)]=4
 label=np.array(label,dtype=np.uint8)
 output = cv2.imread(labels_index, 0)
 acc, class_acc, miou, fiou = evlate(output, label, class_num=5)
 mean_miou=np.mean(miou)
 print(acc,class_acc)
 with open('./result/svm.txt','a',encoding='utf-8') as f:
         f.writelines(img_index+""+'acc '+str(acc)+" "+'class_acc '+str(np.mean(class_acc[np.nonzero(class_acc)]))+" "+"iou[{},{},{},{},{}] ".format(miou[0],miou[1],miou[2],miou[3],miou[4])+" "+"miou "+str(np.mean(miou))+"fiou "+str(fiou))




# acc, class_acc, miou, fiou = evlate(output/255, label/255, class_num=2)
# mean_miou=np.mean(miou)
# print("acc",acc,"class_acc",class_acc,"iou",miou,"m_iou",mean_miou,"fiou",fiou)
# palette = {(255, 255, 255): 3,
#            (255, 0, 0): 1,
#            (128, 128, 128): 0,
#            (0, 255, 0): 2,
#            }
# output=np.array(output /255,dtype=np.uint8)
# new_img = np.zeros((256, 256, 3), dtype=np.uint8)
# for i in range(output.shape[0]):
#     for j in range(output.shape[1]):
#         for c, index_label in palette.items():
#             if (output[i, j] == index_label):
#                 for index in range(len(c)):
#                     new_img[i, j, index] = c[index]
# cv2.imwrite('./paper_result/svm/label/9.png', new_img)
# cv2.imshow('lable',new_img)
# output=np.array(label /255,dtype=np.uint8)
# new_img = np.zeros((256, 256, 3), dtype=np.uint8)
# for i in range(output.shape[0]):
#     for j in range(output.shape[1]):
#         for c, index_label in palette.items():
#             if (output[i, j] == index_label):
#                 for index in range(len(c)):
#                     new_img[i, j, index] = c[index]
# cv2.imwrite('./paper_result/svm/predict/9.png', new_img)
# cv2.imshow('output',new_img)
# cv2.imshow('img',img)
# cv2.waitKey(0)
