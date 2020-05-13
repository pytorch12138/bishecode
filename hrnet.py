import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import cv2
class basic_block(nn.Module):
    def __init__(self,inplanes,planes,short_cut=False):
        super(basic_block, self).__init__()
        self.short_cut=short_cut
        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv=nn.Conv2d(inplanes,planes,kernel_size=1,stride=1,bias=False)
        self.bn=nn.BatchNorm2d(planes)
    def forward(self, input):
        out=self.conv1(input)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)
        if self.short_cut:
            residual=self.conv(input)
            residual=self.bn(residual)
            out=torch.add(residual,out)
        else:
             out=torch.add(out,input)
        out=self.relu(out)
        return out
class bottle_block(nn.Module):
    def __init__(self,inplanes,outplanes,short_cut=False):
        super(bottle_block, self).__init__()
        planes=int(outplanes/4)
        self.short_cut=short_cut
        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn2= nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.conv=nn.Conv2d(inplanes, planes*4, kernel_size=1, stride=1, bias=False)
        self.bn=nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=False)
    def forward(self, input):
        out=self.conv1(input)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)
        out=self.conv3(out)
        out=self.bn3(out)
        out=self.relu(out)
        if self.short_cut:
            residual=self.conv(input)
            residual=self.bn(residual)
            out=torch.add(out,residual)
        else:
            out=torch.add(out,input)
        out=self.relu(out)
        return  out
class stem_net(nn.Module):
    def __init__(self,inplanes,planes):
        super(stem_net, self).__init__()
        self.conv=nn.Conv2d(in_channels=inplanes,out_channels=planes,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn=nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.bottle_net1=bottle_block(planes,planes*2,short_cut=True)
        self.bottle_net2=bottle_block(planes*2,planes*2,short_cut=False)
        self.bottle_net3 = bottle_block(planes*2, planes*2 , short_cut=False)
        self.bottle_net4 = bottle_block(planes*2,planes*4, short_cut=True)
    def forward(self, input):
        out=self.conv(input)
        out=self.bn(out)
        out=self.relu(out)
        out=self.bottle_net1(out)
        out=self.bottle_net2(out)
        out=self.bottle_net3(out)
        out=self.bottle_net4(out)
        return  out
class final_layer(nn.Module):
    def __init__(self,planes,class_num=1):
        super(final_layer, self).__init__()
        self.unsample1=nn.UpsamplingBilinear2d(scale_factor=2)
        self.unsample2=nn.UpsamplingBilinear2d(scale_factor=4)
        self.unsample3=nn.UpsamplingBilinear2d(scale_factor=8)
        self.conv = nn.Conv2d(int(planes/2), planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.conv1=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes*2, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes * 4, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.unsample4=nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv4=nn.Conv2d(planes*4,class_num,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn4=nn.BatchNorm2d(class_num)

    def forward(self, input1,input2,input3,input4):
        out1=self.conv(input1)
        out1=self.bn(out1)
        out2=self.conv1(input2)
        out2=self.bn1(out2)
        out2=self.unsample1(out2)
        out3 = self.conv2(input3)
        out3 = self.bn2(out3)
        out3 = self.unsample2(out3)
        out4 = self.conv3(input4)
        out4 = self.bn3(out4)
        out4 = self.unsample3(out4)
        out=torch.cat([out1,out2,out3,out4],dim=1)
        out=self.unsample4(out)
        out=self.conv4(out)
        return out
class Model(nn.Module):
    def __init__(self,inplanes,planes,class_num=1):
        super(Model, self).__init__()
        self.out_layer_list = [int(planes/2),planes,planes*2,planes*4]
        self.stem_net=stem_net(inplanes,int(planes/2))
        # 第一部分
        self.transition_layer1_1=nn.Sequential(nn.Conv2d(planes*2,self.out_layer_list[0],kernel_size=3, stride=1, padding=1,bias=False),nn.BatchNorm2d(self.out_layer_list[0]),nn.LeakyReLU())
        self.transition_layer1_2 = nn.Sequential(
            nn.Conv2d(planes * 2, self.out_layer_list[1], kernel_size=3, stride=2,padding=1, bias=False),
            nn.BatchNorm2d(self.out_layer_list[1]), nn.LeakyReLU())
        self.make_branch1_1=nn.Sequential(basic_block(self.out_layer_list[0],self.out_layer_list[0],False),basic_block(self.out_layer_list[0],self.out_layer_list[0],False),basic_block(self.out_layer_list[0],self.out_layer_list[0],False),basic_block(self.out_layer_list[0],self.out_layer_list[0],False))
        self.make_branch1_2 = nn.Sequential(basic_block(self.out_layer_list[1], self.out_layer_list[1], False),basic_block(self.out_layer_list[1], self.out_layer_list[1], False),basic_block(self.out_layer_list[1], self.out_layer_list[1], False),basic_block(self.out_layer_list[1], self.out_layer_list[1], False))
        self.fuse_layer1_1=nn.Sequential(nn.Conv2d(self.out_layer_list[1],self.out_layer_list[0],kernel_size=1,stride=1,bias=False),nn.BatchNorm2d(self.out_layer_list[0]),nn.LeakyReLU(),nn.UpsamplingBilinear2d(scale_factor=2))
        self.fuse_layer1_2=nn.Sequential(nn.Conv2d(self.out_layer_list[0],self.out_layer_list[1],kernel_size=3,stride=2,padding=1,bias=False),nn.BatchNorm2d(self.out_layer_list[1]))
        # 第二部分
        self.transition_layer2_1=nn.Sequential(nn.Conv2d(self.out_layer_list[0],self.out_layer_list[0],kernel_size=3, stride=1, padding=1,bias=False),nn.BatchNorm2d(self.out_layer_list[0]),nn.LeakyReLU())
        self.transition_layer2_2 = nn.Sequential(
        nn.Conv2d(self.out_layer_list[1], self.out_layer_list[1], kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[1]), nn.LeakyReLU())
        self.transition_layer2_3 = nn.Sequential(
            nn.Conv2d(self.out_layer_list[1], self.out_layer_list[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.out_layer_list[2]), nn.LeakyReLU())
        self.make_branch2_1 = nn.Sequential(basic_block(self.out_layer_list[0], self.out_layer_list[0], False), basic_block(self.out_layer_list[0], self.out_layer_list[0], False),basic_block(self.out_layer_list[0], self.out_layer_list[0], False),basic_block(self.out_layer_list[0], self.out_layer_list[0], False))
        self.make_branch2_2=nn.Sequential(basic_block(self.out_layer_list[1], self.out_layer_list[1], False),basic_block(self.out_layer_list[1], self.out_layer_list[1], False),basic_block(self.out_layer_list[1], self.out_layer_list[1], False),basic_block(self.out_layer_list[1], self.out_layer_list[1], False))
        self.make_branch2_3=nn.Sequential(basic_block(self.out_layer_list[2], self.out_layer_list[2], False),basic_block(self.out_layer_list[2], self.out_layer_list[2], False),basic_block(self.out_layer_list[2], self.out_layer_list[2], False),basic_block(self.out_layer_list[2], self.out_layer_list[2], False))
    #     融合层
        self.fuse_layer2_1_1=nn.Sequential(nn.Conv2d(self.out_layer_list[1],self.out_layer_list[0],kernel_size=1,stride=1,bias=False),nn.BatchNorm2d(self.out_layer_list[0]),nn.UpsamplingBilinear2d(scale_factor=2))
        self.fuse_layer2_1_2 = nn.Sequential(
            nn.Conv2d(self.out_layer_list[2], self.out_layer_list[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.out_layer_list[0]), nn.UpsamplingBilinear2d(scale_factor=4))
        self.fuse_layer2_2_1=nn.Sequential( nn.Conv2d(self.out_layer_list[0], self.out_layer_list[1], kernel_size=3, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(self.out_layer_list[1]))
        self.fuse_layer2_2_2 = nn.Sequential(
            nn.Conv2d(self.out_layer_list[2], self.out_layer_list[1], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.out_layer_list[1]), nn.UpsamplingBilinear2d(scale_factor=2))
        self.fuse_layer2_3_1 = nn.Sequential(
        nn.Conv2d(self.out_layer_list[0], self.out_layer_list[2], kernel_size=3, stride=2, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[2]),nn.Conv2d(self.out_layer_list[2], self.out_layer_list[2], kernel_size=3, stride=2, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[2]))
        self.fuse_layer2_3_2 = nn.Sequential(nn.Conv2d(self.out_layer_list[1], self.out_layer_list[2], kernel_size=3, stride=2,padding=1, bias=False),
        nn.BatchNorm2d(self.out_layer_list[2]))
    # 第三部分
        self.transition_layer3_1=nn.Sequential(nn.Conv2d(self.out_layer_list[0],self.out_layer_list[0],kernel_size=3, stride=1, padding=1,bias=False),nn.BatchNorm2d(self.out_layer_list[0]),nn.LeakyReLU())
        self.transition_layer3_2=nn.Sequential(
        nn.Conv2d(self.out_layer_list[1], self.out_layer_list[1], kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[1]), nn.LeakyReLU())
        self.transition_layer3_3=nn.Sequential(
        nn.Conv2d(self.out_layer_list[2], self.out_layer_list[2], kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[2]), nn.LeakyReLU())
        self.transition_layer3_4 = nn.Sequential(
            nn.Conv2d(self.out_layer_list[2], self.out_layer_list[3], kernel_size=3, stride=2, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[3]), nn.LeakyReLU())
        self.make_branch3_1 = nn.Sequential(basic_block(self.out_layer_list[0], self.out_layer_list[0], False),basic_block(self.out_layer_list[0], self.out_layer_list[0], False),basic_block(self.out_layer_list[0], self.out_layer_list[0], False),basic_block(self.out_layer_list[0], self.out_layer_list[0], False))
        self.make_branch3_2 = nn.Sequential(basic_block(self.out_layer_list[1], self.out_layer_list[1], False),basic_block(self.out_layer_list[1], self.out_layer_list[1], False),basic_block(self.out_layer_list[1], self.out_layer_list[1], False),basic_block(self.out_layer_list[1], self.out_layer_list[1], False))
        self.make_branch3_3 = nn.Sequential(basic_block(self.out_layer_list[2], self.out_layer_list[2], False), basic_block(self.out_layer_list[2], self.out_layer_list[2], False),basic_block(self.out_layer_list[2], self.out_layer_list[2], False),basic_block(self.out_layer_list[2], self.out_layer_list[2], False))
        self.make_branch3_4 = nn.Sequential(basic_block(self.out_layer_list[3], self.out_layer_list[3], False),basic_block(self.out_layer_list[3], self.out_layer_list[3], False),basic_block(self.out_layer_list[3], self.out_layer_list[3], False),basic_block(self.out_layer_list[3], self.out_layer_list[3], False))
        self.fuse_layer3_1_1=nn.Sequential(nn.Conv2d(self.out_layer_list[1],self.out_layer_list[0],kernel_size=1,stride=1,bias=False),nn.BatchNorm2d(self.out_layer_list[0]),nn.UpsamplingBilinear2d(scale_factor=2))
        self.fuse_layer3_1_2=nn.Sequential(nn.Conv2d(self.out_layer_list[2], self.out_layer_list[0], kernel_size=1, stride=1, bias=False),nn.BatchNorm2d(self.out_layer_list[0]), nn.UpsamplingBilinear2d(scale_factor=4))
        self.fuse_layer3_1_3=nn.Sequential(nn.Conv2d(self.out_layer_list[3], self.out_layer_list[0], kernel_size=1, stride=1, bias=False),nn.BatchNorm2d(self.out_layer_list[0]), nn.UpsamplingBilinear2d(scale_factor=8))
        self.fuse_layer3_2_1=nn.Sequential( nn.Conv2d(self.out_layer_list[0], self.out_layer_list[1], kernel_size=3, stride=2,padding=1,bias=False),nn.BatchNorm2d(self.out_layer_list[1]))
        self.fuse_layer3_2_2 = nn.Sequential(nn.Conv2d(self.out_layer_list[2], self.out_layer_list[1], kernel_size=1, stride=1,bias=False), nn.BatchNorm2d(self.out_layer_list[1]),nn.UpsamplingBilinear2d(scale_factor=2))
        self.fuse_layer3_2_3 = nn.Sequential( nn.Conv2d(self.out_layer_list[3], self.out_layer_list[1], kernel_size=1, stride=1,bias=False),nn.BatchNorm2d(self.out_layer_list[1]), nn.UpsamplingBilinear2d(scale_factor=4))

        self.fuse_layer3_3_1 = nn.Sequential( nn.Conv2d(self.out_layer_list[0], self.out_layer_list[2], kernel_size=3, stride=2, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[2]), nn.Conv2d(self.out_layer_list[2], self.out_layer_list[2], kernel_size=3, stride=2, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[2]))
        self.fuse_layer3_3_2 = nn.Sequential( nn.Conv2d(self.out_layer_list[1], self.out_layer_list[2], kernel_size=3, stride=2, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[2]))
        self.fuse_layer3_3_3 = nn.Sequential(nn.Conv2d(self.out_layer_list[3], self.out_layer_list[2], kernel_size=1, stride=1, bias=False),nn.BatchNorm2d(self.out_layer_list[2]),nn.UpsamplingBilinear2d(scale_factor=2))

        self.fuse_layer3_4_1=nn.Sequential(nn.Conv2d(self.out_layer_list[0], self.out_layer_list[3], kernel_size=3, stride=2, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[3]), nn.Conv2d(self.out_layer_list[3], self.out_layer_list[3], kernel_size=3, stride=2, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[3]), nn.Conv2d(self.out_layer_list[3], self.out_layer_list[3], kernel_size=3, stride=2, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[3]))
        self.fuse_layer3_4_2=nn.Sequential(nn.Conv2d(self.out_layer_list[1], self.out_layer_list[3], kernel_size=3, stride=2, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[3]), nn.Conv2d(self.out_layer_list[3], self.out_layer_list[3], kernel_size=3, stride=2, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[3]))
        self.fuse_layer3_4_3 = nn.Sequential(nn.Conv2d(self.out_layer_list[2], self.out_layer_list[3], kernel_size=3, stride=2, padding=1, bias=False),nn.BatchNorm2d(self.out_layer_list[3]))
        self.final_layer=final_layer(planes=planes,class_num=class_num)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, input):
         out_input=self.stem_net(input)
         out11=self.transition_layer1_1(out_input)
         out12=self.transition_layer1_2(out_input)
         out11=self.make_branch1_1(out11)
         out12=self.make_branch1_2(out12)
         out11_f=self.fuse_layer1_1(out12)
         out21=torch.add(out11,out11_f)
         out12_f=self.fuse_layer1_2(out11)
         out22=torch.add(out12,out12_f)
         out21=self.transition_layer2_1(out21)
         out22=self.transition_layer2_2(out22)
         out23=self.transition_layer2_3(out22)
         out21=self.make_branch2_1(out21)
         out22=self.make_branch2_2(out22)
         out23=self.make_branch2_3(out23)
         out21_1f=self.fuse_layer2_1_1(out22)
         out21_2f=self.fuse_layer2_1_2(out23)
         out31=out21+out21_1f+out21_2f
         out22_1f=self.fuse_layer2_2_1(out21)
         out22_2f=self.fuse_layer2_2_2(out23)
         out32=out22+out22_1f+out22_2f
         out23_1f=self.fuse_layer2_3_1(out21)
         out23_2f = self.fuse_layer2_3_2(out22)
         out33=out23+out23_2f+out23_1f
         out31=self.transition_layer3_1(out31)
         out32=self.transition_layer3_2(out32)
         out33=self.transition_layer3_3(out33)
         out34=self.transition_layer3_4(out33)
         out31=self.make_branch3_1(out31)
         out32 = self.make_branch3_2(out32)
         out33 = self.make_branch3_3(out33)
         out34 = self.make_branch3_4(out34)
         out31_1f=self.fuse_layer3_1_1(out32)
         out31_2f=self.fuse_layer3_1_2(out33)
         out31_3f=self.fuse_layer3_1_3(out34)
         out41=out31+out31_1f+out31_2f+out31_3f
         out32_1f=self.fuse_layer3_2_1(out31)
         out32_2f=self.fuse_layer3_2_2(out33)
         out32_3f = self.fuse_layer3_2_3(out34)
         out42=out32+out32_1f+out32_2f+out32_3f
         out33_1f = self.fuse_layer3_3_1(out31)
         out33_2f = self.fuse_layer3_3_2(out32)
         out33_3f = self.fuse_layer3_3_3(out34)
         out43 = out33 + out33_1f + out33_2f + out33_3f
         out34_1f = self.fuse_layer3_4_1(out31)
         out34_2f = self.fuse_layer3_4_2(out32)
         out34_3f = self.fuse_layer3_4_3(out33)
         out44 = out34+ out34_1f + out34_2f + out34_3f
         out=self.final_layer(out41,out42,out43,out44)
         return out
import os
from evaluate import evlate
img_list=os.listdir('./house/test/house/imgs/')
imgs_list=[]
labels_list=[]
for img_folder in img_list:
 imgs_list.append('./house/test/house/imgs/'+img_folder)
 labels_list.append('./house/test/house/labels/'+img_folder)
torch.backends.cudnn.benchmark = True
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(inplanes=3, planes=96, class_num=1).to(device)
# model.load_state_dict(torch.load( './model//model.pth',)['state'])

model.load_state_dict(torch.load( './model/house_hrent48/model_96_198.pth',map_location='cpu')['state'])
model.eval()
# example = torch.rand(1, 3, 256, 256)
# example = Variable(example.to(device))
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("./pt/hr_house_48_model.pt")





simoid=nn.Sigmoid()
img_index='{}.png'.format(15)
img=cv2.imread('./house/train/house/imgs/'+img_index)
label=cv2.imread('./house/train/house/labels/'+img_index,0)
# print(label)
# label=np.where(label==4,np.ones_like(label),np.zeros_like(label))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img1=img/255.0
img_x=torch.from_numpy(img1).float()
img_x=img_x.permute(2,0,1)
img_x=torch.stack([img_x])
imgs = Variable(img_x.to(device))
output=model(imgs)
output=simoid(output)
c=output.size()
output = output.view(c[0] * c[1], c[2], c[3])
output = output.detach().cpu().numpy()
output = np.where(output >= 0.5, 0, 1)
output = np.array(output)[0]
output=output
output = np.array(output, dtype=np.uint8)
palette = {(0, 255, 255): 0,
               (64,64,64):1,
               }
    # output=np.array(output/255,dtype=np.uint8)
new_img = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            for c, index_label in palette.items():
                if (output[i, j] == index_label):
                        for index in range(len(c)):
                            new_img[i, j, index] = c[index]
# cv2.imwrite('./paper_result/hrnet48/predict/'+img_index,new_img)
cv2.imshow("output",new_img)
output = label
palette = {
               (0, 255, 255): 0,
               (64,64,64):1,
               }
new_img = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            for c, index_label in palette.items():
                if (output[i, j] == index_label):
                        for index in range(len(c)):
                            new_img[i, j, index] = c[index]
# cv2.imwrite('./paper_result/hrnet48/labels/'+img_index,new_img)
cv2.imshow("label",new_img)
print("预测完毕")
cv2.waitKey(0)



# # model.load_state_dict(torch.load( './model/river_hrnet/model.pth',)['state'])
# model.load_state_dict(torch.load( './model/house_hrnet/hr_net_house.pth',)['state'])
# model.eval()
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
#
#














# img_code="58.png"
# img_folder='./house/test/house/imgs/'+img_code
# label_folder='./house/test/house/labels/'+img_code
# predict_folder='./predict/hrnet/'+img_code
# label_='./label/'+img_code
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
# labels_img=labels_img*255
# cv2.imwrite(label_,labels_img)
# cv2.imshow("out",output5)
# cv2.imshow("img",img)
# cv2.imshow("labels",labels_img)
# cv2.waitKey(0)
#




# model.eval()
# img_index=0
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
#     label5=label
#     acc,class_acc,miou,fiou=evlate(output,label5,class_num=2)
#     acc_list.append(acc)
#     miou_list.append(miou)
#     fiou_list.append(fiou)
# mean_acc = np.mean(acc_list)
# mean_miou = np.mean(miou)
# mean_fiou = np.mean(fiou_list)
# print("本次预测样本为200", "平均准确率为{}".format(mean_acc), "平均iou为{}".format(mean_miou), "平均fiou为{}".format(mean_fiou))
# #     with open('./hr_river_evlate_test.txt','a',encoding='utf-8') as f:
# #         f.writelines('测试图片为{}.png'.format(img_index)+' 准确率:{}'.format(acc)+' 类别准确度为[{},{}]'.format(class_acc[0],class_acc[1])+' 平均iou为{}'.format(miou)+' 频权iou为{}'.format(fiou))
# #  else:
# #      break
#  # cv2.imshow("out",output5)
#  # cv2.waitKey(0)


