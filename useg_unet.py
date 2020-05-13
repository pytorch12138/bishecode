import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
torch.set_printoptions(threshold=300000)
from torch.autograd import Variable
class Attention_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Attention_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                                  nn.BatchNorm2d(out_ch), nn.Sigmoid())
        self.psi = nn.Sequential(nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(1),
                                 nn.Sigmoid())
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.conv(g)
        x1 = self.conv(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
class Segnet_unet(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(Segnet_unet, self).__init__()
        #模块1
        self.conv11=nn.Conv2d(input_nbr,32,kernel_size=3,padding=1)
        self.bn11=nn.BatchNorm2d(32,momentum=0.1)
        self.conv12=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.bn12=nn.BatchNorm2d(64,momentum=0.1)
        self.conv_u1=nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.bn_u1=nn.BatchNorm2d(64,momentum=0.1)

       #模块2
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=0.1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=0.1)
        self.conv_u2= nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_u2 = nn.BatchNorm2d(128, momentum=0.1)
        #模块3
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=0.1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 =  nn.BatchNorm2d(256, momentum=0.1)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=0.1)
        #模块4
        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=0.1)
        #模块5
        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=0.1)
        # 模块6
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=0.1)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=0.1)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=0.1)
        # 模块7
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=0.1)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=0.1)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=0.1)
        # 模块8
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=0.1)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=0.1)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=0.1)
        # 模块9
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=0.1)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=0.1)
        # 模块10
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=0.1)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)
        self.relu=nn.ReLU(inplace=True)
        #注意力模块
        self.at9=Attention_block(in_ch=128,out_ch=64)
        self.at10 = Attention_block(in_ch=64, out_ch=32)
    def forward(self, input):
         #模块1
         out11=self.conv11(input)
         out11=self.bn11(out11)
         out11=self.relu(out11)
         out12=self.conv12(out11)
         out12=self.bn12(out12)
         out12=self.relu(out12)
         out1p, outid1 = F.max_pool2d(out12, kernel_size=2, stride=2, return_indices=True)
         #模块2
         out21 = self.conv21(out1p)
         out21 = self.bn21(out21)
         out21 = self.relu(out21)
         out22 = self.conv22(out21)
         out22 = self.bn22(out22)
         out22 = self.relu(out22)
         out2p, outid2 = F.max_pool2d(out22, kernel_size=2, stride=2, return_indices=True)
         #模块3
         out31 = self.conv31(out2p)
         out31 = self.bn31(out31)
         out31 = self.relu(out31)
         out32 = self.conv32(out31)
         out32 = self.bn32(out32)
         out32 = self.relu(out32)
         out33 = self.conv33(out32)
         out33 = self.bn33(out33)
         out33 = self.relu(out33)
         out3p, outid3 = F.max_pool2d(out33, kernel_size=2, stride=2, return_indices=True)
         #模块4
         out41 = self.conv41(out3p)
         out41 = self.bn41(out41)
         out41 = self.relu(out41)
         out42 = self.conv42(out41)
         out42 = self.bn42(out42)
         out42 = self.relu(out42)
         out43 = self.conv43(out42)
         out43 = self.bn43(out43)
         out43 = self.relu(out43)
         out4p, outid4 = F.max_pool2d(out43, kernel_size=2, stride=2, return_indices=True)

         #模块5
         out51 = self.conv51(out4p)
         out51 = self.bn51(out51)
         out51 = self.relu(out51)
         out52 = self.conv52(out51)
         out52 = self.bn52(out52)
         out52 = self.relu(out52)
         out53 = self.conv53(out52)
         out53 = self.bn53(out53)
         out53 = self.relu(out53)
         out5p, outid5 = F.max_pool2d(out53, kernel_size=2, stride=2, return_indices=True)
         #模块6
         out5d=F.max_unpool2d(out5p,outid5,kernel_size=2,stride=2)
         out53d=self.conv53d(out5d)
         out53d=self.bn53d(out53d)
         out53d=self.relu(out53d)
         out52d = self.conv52d(out53d)
         out52d = self.bn52d(out52d)
         out52d = self.relu(out52d)
         out51d = self.conv51d(out52d)
         out51d = self.bn51d(out51d)
         out51d = self.relu(out51d)
         #模块7
         out4d = F.max_unpool2d(out51d, outid4, kernel_size=2, stride=2)
         out63d = self.conv43d(out4d)
         out63d = self.bn43d(out63d)
         out63d = self.relu(out63d)
         out62d = self.conv42d(out63d)
         out62d = self.bn42d(out62d)
         out62d = self.relu(out62d)
         out61d = self.conv41d(out62d)
         out61d = self.bn41d(out61d)
         out61d = self.relu(out61d)
         #模块8
         out3d = F.max_unpool2d(out61d, outid3, kernel_size=2, stride=2)
         out73d = self.conv33d(out3d)
         out73d = self.bn33d(out73d)
         out73d = self.relu(out73d)
         out72d = self.conv32d(out73d)
         out72d = self.bn32d(out72d)
         out72d = self.relu(out72d)
         out71d = self.conv31d(out72d)
         out71d = self.bn31d(out71d)
         out71d = self.relu(out71d)
        #模块9
         out2d = F.max_unpool2d(out71d, outid2, kernel_size=2, stride=2)
         out82d=self.conv22d(out2d)
         out82d=self.bn22d(out82d)
         out82d=self.relu(out82d)
         out22=self.at9(out82d,out22)
         merge9 = torch.cat([out82d, out22], dim=1)
         out_2d=self.conv_u2(merge9)
         out_2d=self.bn_u2(out_2d)
         out_2d=self.relu(out_2d)
         out81d=self.conv21d(out_2d)
         out1d = F.max_unpool2d(out81d, outid1, kernel_size=2, stride=2)
         out92d=self.conv12d(out1d)
         out92d=self.bn12d(out92d)
         out92d=self.relu(out92d)
         out12 = self.at10(out92d, out12)
         merge10 = torch.cat([out92d, out12], dim=1)
         out_1d = self.conv_u1(merge10)
         out_1d = self.bn_u1(out_1d)
         out_1d = self.relu(out_1d)
         out=self.conv11d(out_1d)
         return out
import time
start_time=time.time()
torch.backends.cudnn.benchmark = True
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Segnet_unet(input_nbr=3, label_nbr=5).to(device)
# model.load_state_dict(torch.load( './model/house_usegnet/hr_best_model.pth')['state'])
model.load_state_dict(torch.load( './model/u_segnet/model_208.pth',map_location='cpu')['state'])
model.eval()
example = torch.rand(1, 3, 256, 256)
example = Variable(example.to(device))
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("./pt/usegmnet_model.pt")



# model.load_state_dict(torch.load( './model/u_segnet/model_208.pth',)['state'])
# model.eval()
# model.eval()
# from thop import profile
# from thop import clever_format
# input = torch.randn(1, 3, 256, 256)
# input = Variable(input.to(device))
# flops, params = profile(model, inputs=(input, ))
# flops, params = clever_format([flops, params], "%.3f")
# end_time=time.time()
# print("flops",flops,"params",params,"time",end_time-start_time)




# softmax=nn.Softmax(dim=1)
# img_index='{}.png'.format(8)
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
# cv2.imwrite('./paper_result/usegnet/predict/'+img_index,new_img)
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
# cv2.imwrite('./paper_result/usegnet/labels/'+img_index,new_img)
# cv2.imshow("label",new_img)
# print("预测完毕")
# cv2.waitKey(0)



# simoid=nn.Sigmoid()
# # img=cv2.imread('./river_test/imgs/0.png')
# img=cv2.imread('./house/test/house/imgs/0.png')
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
# # label=cv2.imread('./river_test/labels/0.png')
# label=cv2.imread('./house/test/house/labels/0.png')
# label=np.array(label*255,dtype=np.uint8)
# cv2.imshow('label',label)
# cv2.waitKey(0)
#






# model.eval()
# img_index=0
# acc_list=[]
# miou_list=[]
# # fiou_list=[]
# # img_index=0
# # acc_list=[]
# # miou_list=[]
# # fiou_list=[]
# # img_code="41.png"
# # img_folder='./house/test/house/imgs/'+img_code
# # label_folder='./house/test/house/labels/'+img_code
# # predict_folder='./predict/u-segnet/'+img_code
# # img=cv2.imread(img_folder)
# # img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # img1=img1/255.0
# # img_x=torch.from_numpy(img1).float()
# # img_x=img_x.permute(2,0,1)
# # img_x=torch.stack([img_x])
# # imgs = Variable(img_x.to(device))
# # output=model(imgs)
# # simoid=nn.Sigmoid()
# # c = output.size()
# # output=simoid(output)
# # c=output.size()
# # output = output.view(c[0] * c[1], c[2], c[3])
# # output = output.detach().cpu().numpy()
# # output = np.where(output > 0.5, 0, 1)
# # output = np.array(output,dtype=np.uint8)
# # output5=output[0]*255
# # cv2.imwrite(predict_folder,output5)
# # labels_img=cv2.imread(label_folder)
# # cv2.imshow("out",output5)
# # cv2.imshow("img",img)
# # cv2.imshow("labels",labels_img*255)
# # cv2.waitKey(0)


# import os
# import cv2
# import numpy as np
# from evaluate import evlate
# img_list=os.listdir('./river_test/imgs/')
# imgs_list=[]
# labels_list=[]
# for img_folder in img_list:
#  imgs_list.append('./river_test/imgs/'+img_folder)
#  labels_list.append('./river_test/labels/'+img_folder)
# torch.backends.cudnn.benchmark = True
# cuda = True if torch.cuda.is_available() else False
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Segnet_unet(input_nbr=3, label_nbr=1).to(device)
# model.load_state_dict(torch.load( './model/river_usegment/model.pth',)['state'])
# model.eval()
# img_index=0
# model.eval()
# img_index=0
# acc_list=[]
# miou_list=[]
# fiou_list=[]
# for (img_folder,label_folder) in zip(imgs_list,labels_list):
#  if img_index<200:
#     print(img_folder)
#     img_index+=1
#     img=cv2.imread(img_folder)
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
#     label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
#     output=output5/255.0
#     label5=label/255.0
#     acc,class_acc,miou,fiou=evlate(output,label5,class_num=2)
#     acc_list.append(acc)
#     miou_list.append(miou)
#     fiou_list.append(fiou)
# mean_acc = np.mean(acc_list)
# mean_miou = np.mean(miou)
# mean_fiou = np.mean(fiou_list)
# print("本次预测样本为200", "平均准确率为{}".format(mean_acc), "平均iou为{}".format(mean_miou), "平均fiou为{}".format(mean_fiou))
#
# #     with open('./usegnet_river_evlate.txt','a',encoding='utf-8') as f:
#         f.writelines('测试图片为{}.png'.format(img_index)+' 准确率:{}'.format(acc)+' 类别准确度为[{},{}]'.format(class_acc[0],class_acc[1])+' 平均iou为{}'.format(miou)+' 频权iou为{}'.format(fiou))
#  else:
#      break
#  # cv2.imshow("img",output5)
#  # cv2.imshow("label",label5)
#  # cv2.waitKey(0)