import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class fuser_layer(nn.Module):
    def __init__(self,in_ch,out_ch):
         super(fuser_layer, self).__init__()
         self.conv1=nn.Conv2d(in_ch,out_ch*16,kernel_size=3, stride=1, padding=1)
         self.bn1=nn.BatchNorm2d(out_ch*16)
         self.conv2 = nn.Conv2d(out_ch*16, out_ch, kernel_size=3, stride=1, padding=1)
         self.relu=nn.ReLU(inplace=True)
         self.sigmoid=nn.Sigmoid()
    def forward(self, input):
         out=self.conv1(input)
         out=self.bn1(out)
         out=self.relu(out)
         out=self.conv2(out)
         out=self.sigmoid(out)
         return out
class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6 = DoubleConv(512, 256)
        self.up7=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv7=DoubleConv(256,128)
        self.up8= nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8=DoubleConv(64,64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = DoubleConv(32, 32)
        self.conv11 = nn.Conv2d(32,out_ch, 1)
    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        c6_1=self.conv6(up_6)
        merge6 = torch.cat([c6_1,c4], dim=1)
        c6_2=self.conv6(merge6)
        up_7=self.up7(c6_2)
        c7_1=self.conv7(up_7)
        merge7 = torch.cat([c7_1, c3], dim=1)
        c7_2=self.conv7(merge7)
        up_8=self.up8(c7_2)
        c8_1=self.conv8(up_8)
        merge8 = torch.cat([c8_1, c2], dim=1)
        c9=self.conv9(merge8)
        up_9=self.up9(c9)
        c10=self.conv10(up_9)
        c11= self.conv11(c10)
        out=c11
        return out
import numpy as np
torch.set_printoptions(threshold=300000)
from torch.autograd import Variable
import cv2
import os
import time
from evaluate import evlate
img_list=os.listdir('./house/test/house/imgs/')
imgs_list=[]
labels_list=[]
for img_folder in img_list:
 imgs_list.append('./house/test/house/imgs/'+img_folder)
 labels_list.append('./house/test/house/labels/'+img_folder)
start_time=time.time()
torch.backends.cudnn.benchmark = True
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet(in_ch=3, out_ch=5).to(device)
model.load_state_dict(torch.load( './model/unet/u_model_320.pth',map_location='cpu')['state'])
model.eval()
example = torch.rand(1, 3, 256, 256)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("./pt/unet_model.pt")






# from thop import profile
# from thop import clever_format
# input = torch.randn(1, 3, 256, 256)
# input = Variable(input.to(device))
# flops, params = profile(model, inputs=(input, ))
# flops, params = clever_format([flops, params], "%.3f")
# end_time=time.time()
# print("flops",flops,"params",params,"time",end_time-start_time)












# # model.load_state_dict(torch.load( './model/segnet/s_model_240.pth')['state'])
# # model.eval()
# softmax=nn.Softmax(dim=1)
# img_list=os.listdir('./test/imgs/')
# img_list.sort(key=lambda x: int(x[:-4]))
# print(img_list)
# imgs_list=[]
# labels_list=[]
# for img_folder in img_list:
#  imgs_list.append('./test/imgs/'+img_folder)
#  labels_list.append('./test/label/'+img_folder)
# for (img_index,labels_index) in zip(imgs_list,labels_list):
#      print(img_index)
#      img=cv2.imread(img_index)
#      label=cv2.imread(labels_index,0)
#      img1=img/255
#      img_x=torch.from_numpy(img1).float()
#      img_x=img_x.permute(2,0,1)
#      img_x=torch.stack([img_x])
#      imgs = Variable(img_x.to(device))
#      output=model(imgs)
#      print(output.shape)
#      output=softmax(output)
#      c=output.size()
#      output=torch.argmax(output,dim=1)
#      output = output.detach().cpu().numpy()[0]
#      acc,class_acc,miou,fiou=evlate(output,label,class_num=5)
#      print(miou[0])
#      with open('./result/unet.txt','a',encoding='utf-8') as f:
#          f.writelines(img_index+""+'acc '+str(acc)+" "+'class_acc '+str(np.mean(class_acc[np.nonzero(class_acc)]))+" "+"iou[{},{},{},{},{}] ".format(miou[0],miou[1],miou[2],miou[3],miou[4])+" "+"miou "+str(np.mean(miou))+"fiou "+str(fiou))




# softmax=nn.Softmax(dim=1)
# img_index='{}.png'.format(858)
# img=cv2.imread('./test/imgs/'+img_index)
# label=cv2.imread('./test/label/'+img_index,0)
# # print(label)
# # label=np.where(label==4,np.ones_like(label),np.zeros_like(label))
# img1=img/255
# img_x=torch.from_numpy(img1).float()
# img_x=img_x.permute(2,0,1)
# img_x=torch.stack([img_x])
# imgs = Variable(img_x.to(device))
# output=model(imgs)
# print(output.shape)
# output=softmax(output)
# c=output.size()
# output=torch.argmax(output,dim=1)
# output = output.detach().cpu().numpy()[0]
# print("sss",output)
# # output = np.where(output >= 0.5, 0, 1)
# # output = np.array(output)[0]
# output=output
# output = np.array(output, dtype=np.uint8)
# palette = {(69, 69, 69): 1,
#                (0, 255, 255): 3,
#                (0, 255, 0): 2,
#                (255, 0, 0): 4,
#                (19, 69, 139): 0}
#     # output=np.array(output/255,dtype=np.uint8)
# new_img = np.zeros((256, 256, 3), dtype=np.uint8)
# for i in range(output.shape[0]):
#         for j in range(output.shape[1]):
#             for c, index_label in palette.items():
#                 if (output[i, j] == index_label):
#                         for index in range(len(c)):
#                             new_img[i, j, index] = c[index]
# cv2.imwrite('./paper_result/unet/predict/'+img_index,new_img)
# cv2.imshow("output",new_img)
# output = label
# palette = {(69, 69, 69): 1,
#                (0, 255, 255): 3,
#                (0, 255, 0): 2,
#                (255, 0, 0): 4,
#                (19, 69, 139): 0}
# new_img = np.zeros((256, 256, 3), dtype=np.uint8)
# for i in range(output.shape[0]):
#         for j in range(output.shape[1]):
#             for c, index_label in palette.items():
#                 if (output[i, j] == index_label):
#                         for index in range(len(c)):
#                             new_img[i, j, index] = c[index]
# cv2.imwrite('./paper_result/unet/labels/'+img_index,new_img)
# cv2.imshow("label",new_img)
# print("预测完毕")
# cv2.waitKey(0)










# # img=cv2.imread('./house/test/house/imgs/0.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img1=img/255.0
# img_x=torch.from_numpy(img1).float()
# img_x=img_x.permute(2,0,1)
# img_x=torch.stack([img_x])
# imgs = Variable(img_x.to(device))
# output=model(imgs)
# output=simoid(output)
# c=output.size()
# output = output.view(c[0] * c[1], c[2], c[3])
# output = output.detach().cpu().numpy()
# output = np.where(output >= 0.5, 0, 1)
# output = np.array(output)[0]
# output=output*255
# output = np.array(output, dtype=np.uint8)
# cv2.imshow("img",output)
# label=cv2.imread('./k_test/labels/0.png')
# # label=cv2.imread('./house/test/house/labels/0.png')
# # label=np.array(label*255,dtype=np.uint8)
# cv2.imshow('label',label)
# cv2.waitKey(0)







# img_folder='./house/test/house/imgs/'+img_code
# label_folder='./house/test/house/labels/'+img_code
# predict_folder='./predict/unet/'+img_code
# img=cv2.imread(img_folder)
# img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img1=img1/255.0
# img_x=torch.from_numpy(img1).float()
# img_x=img_x.permute(2,0,1)
# img_x=torch.stack([img_x])
# imgs = Variable(img_x.to(device))
# output=model(imgs)
# simoid=nn.Sigmoid()
# c = output.size()
# output=simoid(output)
# c=output.size()
# output = output.view(c[0] * c[1], c[2], c[3])
# output = output.detach().cpu().numpy()
# output = np.where(output > 0.5, 0, 1)
# output = np.array(output,dtype=np.uint8)
# output5=output[0]*255
# cv2.imwrite(predict_folder,output5)
# labels_img=cv2.imread(label_folder)
# cv2.imshow("out",output5)
# cv2.imshow("img",img)
# cv2.imshow("labels",labels_img*255)
# cv2.waitKey(0)







# img_index=0
# for (img_folder,label_folder) in zip(imgs_list,labels_list):
#  if img_index<200:
#     img_index+=1
#     img=cv2.imread(img_folder)
#     print(img_folder)
#     img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img1=img1/255.0
#     img_x=torch.from_numpy(img1).float()
#     img_x=img_x.permute(2,0,1)
#     img_x=torch.stack([img_x])
#     imgs = Variable(img_x.to(device))
#     output=model(imgs)
#     simoid=nn.Sigmoid()
#     c = output.size()
#     output=simoid(output)
#     c=output.size()
#     output = output.view(c[0] * c[1], c[2], c[3])
#     output = output.detach().cpu().numpy()
#     output = np.where(output > 0.5, 0, 1)
#     output = np.array(output,dtype=np.uint8)
#     output5=output[0]*255
#     label=cv2.imread(label_folder)
#     print(label_folder)
#     label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
#     output=output5/255
#     label=label
#     acc,class_acc,miou,fiou=evlate(output,label,class_num=2)
#     acc_list.append(acc)
#     miou_list.append(miou)
#     fiou_list.append(fiou)
# mean_acc = np.mean(acc_list)
# mean_miou = np.mean(miou)
# mean_fiou = np.mean(fiou_list)
# print("本次预测样本为200", "平均准确率为{}".format(mean_acc), "平均iou为{}".format(mean_miou), "平均fiou为{}".format(mean_fiou))




#     with open('./unet_house_evlate_test.txt','a',encoding='utf-8') as f:
#         f.writelines('测试图片为{}.png'.format(img_index)+' 准确率:{}'.format(acc)+' 类别准确度为[{},{}]'.format(class_acc[0],class_acc[1])+' 平均iou为{}'.format(miou)+' 频权iou为{}'.format(fiou))
#  else:
#      break
#  # cv2.imshow('img',output5)
#  # cv2.imshow('label',label)
#  # cv2.waitKey(0)