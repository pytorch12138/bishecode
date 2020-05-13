import cv2
import matplotlib.pyplot as plt
from evaluate import evlate
import numpy as np
def seg_kmeans_color():
    img = cv2.imread('./test/imgs/7534.png', cv2.IMREAD_COLOR)
    label=cv2.imread('./test/label/858.png',cv2.IMREAD_COLOR)
    lable = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    w,h,_=label.shape
    label=np.zeros(shape=(w,h),dtype=np.uint8)
    for w_index in range(w):
        for h_idnex in range(h):
            if  lable[w_index,h_idnex]==4:
                label[w_index,h_idnex]=0
            if lable[w_index, h_idnex] == 3:
                label[w_index, h_idnex] = 2
            if lable[w_index, h_idnex] == 2:
                label[w_index, h_idnex] = 1
            if lable[w_index, h_idnex] == 0:
                label[w_index, h_idnex] = 3

    # label=np.where(lable==1,np.zeros_like(lable),np.ones_like(lable))
    # label=np.array(lable*51,np.uint8)
    # cv2.imshow("label",label)
    # cv2.waitKey(0)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img_flat = img.reshape((img.shape[0] * img.shape[1], 3))
    img_flat = np.float32(img_flat)

        # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 1000, 0.001)
    flags = cv2.KMEANS_RANDOM_CENTERS
        # 聚类
    compactness, labels, centers = cv2.kmeans(img_flat, 2, None, criteria, 30, flags)
    output = labels.reshape((img.shape[0], img.shape[1]))
    # print(output)
    # output=np.array(output*255,np.uint8)
    # print(label)
    acc, class_acc, miou, fiou = evlate(output, label, class_num=4)
    mean_miou=np.mean(miou)
    print("acc",acc,"class_acc",class_acc,"iou",miou,"m_iou",mean_miou,"fiou",fiou)
    # print(label.shape)
    # print(output.shape)
    # output=np.array(output*51,np.uint8)
    # label=np.array(label*51,np.uint8)
    # cv2.imshow("output",output)
    # cv2.imshow("label", label)
    # cv2.waitKey(0)
    # cv2.imshow("label",label*51)
    palette = {(255, 255, 255): 3,
        (255, 0, 0): 0,
               (0,64,128):1,
               (0, 255, 0): 2,
               }
    # output=np.array(output/255,dtype=np.uint8)
    new_img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            for c, index_label in palette.items():
                if (output[i, j] == index_label):
                        for index in range(len(c)):
                            new_img[i, j, index] = c[index]
    cv2.imwrite('./paper_result/kmeans/predict/7534.png',new_img)
    # cv2.imwrite('./paper_result/kmeans/labels/188.png',new_img)
    cv2.imshow('img',img)
    cv2.imshow("new_img",new_img)
    cv2.waitKey(0)
seg_kmeans_color()
#     # 变换一下图像通道bgr->rgb，否则很别扭啊
#     print(img.shape)
#     b, g, r = cv2.split(img)
#     img = cv2.merge([r, g, b])
#
#     # 3个通道展平
#     img_flat = img.reshape((img.shape[0] * img.shape[1], 3))
#     img_flat = np.float32(img_flat)
#
#     # 迭代参数
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 1000, 0.1)
#     flags = cv2.KMEANS_RANDOM_CENTERS
#
#     # 聚类
#     compactness, labels, centers = cv2.kmeans(img_flat, 2, lable, criteria, 10, flags)
#
#     # 显示结果
#     output = labels.reshape((img.shape[0], img.shape[1]))
#     palette = {(0, 0, 0): 0,
#                (255, 255, 255): 1,
#                (0, 255, 0): 2,
#                (255, 0, 0): 3,
#                (19, 69, 139): 4}
#
#     new_img = np.zeros((256, 256, 3), dtype=np.uint8)
#     for i in range(output.shape[0]):
#         for j in range(output.shape[1]):
#             for c, index_label in palette.items():
#                 if (output[i, j] == index_label):
#                     for index in range(len(c)):
#                         new_img[i, j, index] = c[index]
#     plt.subplot(121), plt.imshow(img), plt.title('input')
#     plt.subplot(122), plt.imshow(new_img), plt.title('kmeans')
#     plt.show()
#
#
# if __name__ == '__main__':
#     seg_kmeans_color()