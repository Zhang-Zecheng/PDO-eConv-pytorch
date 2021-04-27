# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 01:30:46 2020

@author: roderickzzc
"""


import torch
import numpy as np
from torch import nn
from torch.utils import data
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from math import *

def load_rotated_mnist(batch_size):
    #导入数据，将.amat格式转换为numpy array
    data_train = np.loadtxt('C:\\Users\\roderickzzc\\Desktop\\project\\pdo-ecov\\mnist_rotation_new\\mnist_all_rotation_normalized_float_train_valid.amat')
    data_test = np.loadtxt('C:\\Users\\roderickzzc\\Desktop\\project\\pdo-ecov\\mnist_rotation_new\\mnist_all_rotation_normalized_float_test.amat')

    # get train image datas
    x_train_val = data_train[:, :-1] / 1.0
    #由于原始数据集默认为784*1，现改为28*28
    x_train_val=np.reshape(x_train_val,(12000,1,28,28))
    x_test = data_test[:, :-1] / 1.0
    x_test=np.reshape(x_test,(50000,1,28,28))
    # get train image labels
    y_train_val = data_train[:, -1:]
    y_test = data_test[:, -1:]
    #print(x_train_val[0].shape)
    
    # pytorch data loader
    #根据论文抽取2000个样本from training set作为validation
    train_val = torch.utils.data.TensorDataset(torch.Tensor(x_train_val), torch.Tensor(y_train_val))
    train, val = torch.utils.data.random_split(train_val, [10000,2000])
    test = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    ## feature, label = train[0]
    ## print(feature.shape, label) 
    train_iter = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    


    return train_iter, val_iter, test_iter

partial_dict_0 = torch.tensor([[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,1/2,0,0],[0,0,0,0,0],[0,0,-1/2,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,-1/4,0,1/4,0],[0,0,0,0,0],[0,1/4,0,-1/4,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,1,0,0],[0,0,-2,0,0],[0,0,1,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[-1/2,1,0,-1,1/2],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,1/2,-1,1/2,0],[0,0,0,0,0],[0,-1/2,1,-1/2,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1,0,-1,0],[0,-1/2,0,1/2,0],[0,0,0,0,0]],
                    [[0,0,1/2,0,0],[0,0,-1,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,-1/2,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[1,-4,6,-4,1],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[-1/4,1/2,0,-1/2,1/4],[0,0,0,0,0],[1/4,-1/2,0,1/2,-1/4],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,1,-2,1,0],[0,-2,4,-2,0],[0,1,-2,1,0],[0,0,0,0,0]],
                    [[0,-1/4,0,1/4,0],[0,1/2,0,-1/2,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1/4,0,-1/4,0]],
                    [[0,0,1,0,0],[0,0,-4,0,0],[0,0,6,0,0],[0,0,-4,0,0],[0,0,1,0,0]]])
    
p=8
group_angle = [2*k*pi/p+pi/8 for k in range(p)]
tran_to_partial_coef_0 = [torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,cos(x),sin(x),0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,-sin(x),cos(x),0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,0,0,pow(cos(x),2),2*cos(x)*sin(x),pow(sin(x),2),0,0,0,0,0,0,0,0,0],
                                     [0,0,0,-cos(x)*sin(x),pow(cos(x),2)-pow(sin(x),2),sin(x)*cos(x),0,0,0,0,0,0,0,0,0],
                                     [0,0,0,pow(sin(x),2),-2*cos(x)*sin(x),pow(cos(x),2),0,0,0,0,0,0,0,0,0],
                                     [0,0,0,0,0,0,-pow(cos(x),2)*sin(x),pow(cos(x),3)-2*cos(x)*pow(sin(x),2),-pow(sin(x),3)+2*pow(cos(x),2)*sin(x), pow(sin(x),2)*cos(x),0,0,0,0,0],
                                     [0,0,0,0,0,0,cos(x)*pow(sin(x),2),-2*pow(cos(x),2)*sin(x)+pow(sin(x),3),pow(cos(x),3)-2*cos(x)*pow(sin(x),2),sin(x)*pow(cos(x),2),0,0,0,0,0],
                                     [0,0,0,0,0,0,0,0,0,0,pow(sin(x),2)*pow(cos(x),2),-2*pow(cos(x),3)*sin(x)+2*cos(x)*pow(sin(x),3),pow(cos(x),4)-4*pow(cos(x),2)*pow(sin(x),2)+pow(sin(x),4),-2*cos(x)*pow(sin(x),3)+2*pow(cos(x),3)*sin(x),pow(sin(x),2)*pow(cos(x),2)]]) for x in group_angle]

def get_coef(weight,num_inputs,num_outputs):
        #weight.size 1,7,3,3 or 56,7,3,3
        
        transformation = partial_dict_0[[0,1,2,3,4,5,7,8,12],1:4,1:4] #9*3*3
        transformation = transformation.view([9,9])
        inv_transformation = transformation.inverse()#inverse matrix

        betas = torch.reshape(weight,(-1,9))#56*7*9
        #print(betas.size())
        betas = torch.mm(betas,inv_transformation)# 56*7*9
        betas = torch.reshape(betas,(num_inputs,num_outputs,9))
        return betas
    
def z2_kernel(weight,num_inputs,num_outputs,p,partial,tran):
    og_coef = torch.reshape(weight,(num_inputs*num_outputs,9)) #(56*7)*9
    partial_coef = [torch.mm(og_coef,a) for a in tran]#8,(56*7)*15
    partial = torch.reshape(partial,(15,25))#15*25
    kernel = [torch.mm(a,partial) for a in partial_coef]#8,(56*7)*25
    kernel = torch.stack(kernel,dim=1)#(56*7)*8*25
    kernel = torch.reshape(kernel,(num_outputs*p,num_inputs,5,5))#56*56*5*5 or 56*1*5*5
    return kernel

import math
class open_conv2d(nn.Module):
    def __init__(self, num_inputs, num_outputs,p,partial,tran):
        super().__init__()
        self.p=p
        self.num_inputs=num_inputs
        self.num_outputs=num_outputs
        self.partial=partial
        self.tran=tran
        
        self.weight = nn.Parameter(torch.Tensor(self.num_inputs,self.num_outputs,3,3))
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, input):
        #print(self.weight.size())
        betas=get_coef(self.weight,self.num_inputs,self.num_outputs)
        kernel=z2_kernel(betas,self.num_inputs,self.num_outputs,self.p,self.partial,self.tran)
        
        
        
        input_shape = input.size()#input_size: 128,1,h,w & 128,56,h,w
        input = input.view(input_shape[0], self.num_inputs, input_shape[-2], input_shape[-1])
        #y_size: 128,56,h,w
        #print(input.size(),kernel.size())
        outputs = F.conv2d(input, weight=kernel, bias=None, stride=1,
                        padding=1)
        batch_size, _, ny_out, nx_out = outputs.size()
        outputs = outputs.view(batch_size, self.num_outputs*self.p, ny_out, nx_out)
        #y_size: 128,7*8,h,w
        

        return outputs


class g_bn(nn.Module):
    def __init__(self,p):
        super(g_bn, self).__init__()
        self.p=p

    def forward(self, inputs):
        
        channel,height,width = list(inputs.size())[1:]
        inputs = inputs.view(-1,int(channel/p),p,height,width)
        inputs = inputs.view(-1,int(channel/p),height*p,width)
        m = nn.BatchNorm2d(int(channel/p))
        outputs=m(inputs)
        outputs=outputs.view(-1,int(channel/p),p,height,width,)
        outputs = outputs.view(-1,channel,height,width)
        return outputs

class g_conv2d(nn.Module):
    def __init__(self, num_inputs, num_outputs,p,partial,tran):
        super().__init__()
        self.p=p
        self.num_inputs=int(num_inputs/p)
        self.num_outputs=num_outputs
        self.partial=partial
        self.tran=tran
        
        self.weight = nn.Parameter(torch.Tensor(self.p*self.num_inputs,self.num_outputs,3,3))
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, input):
        
        #print(self.weight.size())
        betas=get_coef(self.weight,self.num_inputs*self.p,self.num_outputs)
        og_coef = betas.view(self.num_inputs*self.p*self.num_outputs,9)#(56*7)，9
        tran_to_partial_coef = self.tran #8，9*15
        partial_coef = [torch.mm(og_coef,a) for a in self.tran] #8，（56*7）*15
        
        partial_dict = self.partial
        partial_dict = partial_dict.view(15,25)#15*25
        
        og_kernel_list = [torch.mm(a,partial_dict) for a in partial_coef] #8，（56*7）*25
        og_kernel_list = [og_kernel.view(self.num_inputs,self.p,self.num_outputs,25) for og_kernel in og_kernel_list] #8，（7*8*7*25）
        og_kernel_list = [torch.cat([og_kernel_list[k][:,-k:,:],og_kernel_list[k][:,:-k,:]],dim=1) for k in range(p)] #8，（7*8*7*25）
        
        
        
        
        
        
        kernel = torch.stack(og_kernel_list,dim=3)#7,8,7,8,25
        kernel = kernel.view(self.num_inputs*self.p,self.num_outputs*self.p,5,5)#56,56,5,5
        

        
          
        
        
        
        
        outputs = F.conv2d(input, weight=kernel, bias=None, stride=1,
                        padding=1)
        batch_size, _, ny_out, nx_out = outputs.size()
        outputs = outputs.view(batch_size, self.num_outputs*self.p, ny_out, nx_out)
        #y_size: 128,7*8,h,w
        

        return outputs
    
class P8_PDO_Conv_Z2(open_conv2d):

    def __init__(self, *args, **kwargs):
        super(P8_PDO_Conv_Z2, self).__init__(num_inputs=1, num_outputs=7,p=8,partial=partial_dict_0,tran=tran_to_partial_coef_0)


class P8_PDO_Conv_P8(g_conv2d):

    def __init__(self, *args, **kwargs):
        super(P8_PDO_Conv_P8, self).__init__(num_inputs=56, num_outputs=7,p=8,partial=partial_dict_0,tran=tran_to_partial_coef_0)

class BN_P8(g_bn):
    
    def __init__(self, *args, **kwargs):
        super(BN_P8, self).__init__(p=8)
        

class PDO_eConvs(nn.Module):
    def __init__(self):
        super(PDO_eConvs, self).__init__()
        self.conv1 = P8_PDO_Conv_Z2(1,7,8)
        self.conv2 = P8_PDO_Conv_P8(56,7,8)
        self.conv3 = P8_PDO_Conv_P8(56,7,8)
        self.conv4 = P8_PDO_Conv_P8(56,7,8)
        self.conv5 = P8_PDO_Conv_P8(56,7,8)
        self.conv6 = P8_PDO_Conv_P8(56,7,8)
        self.dropout=nn.Dropout(p=0.2)
        self.bn2 = BN_P8(8)
        self.maxpool2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4*4*7*8, 50)
        self.fc2 = nn.Linear(50, 10)

        
    def forward(self, x):
        x = self.dropout(self.bn2(F.relu(self.conv1(x))))
        print(x.size())
        x = self.maxpool2(self.bn2(F.relu(self.conv2(x))))
        print(x.size())
        x = self.dropout(self.bn2(F.relu(self.conv3(x))))
        print(x.size())
        x = self.dropout(self.bn2(F.relu(self.conv4(x))))
        print(x.size())
        x = self.dropout(self.bn2(F.relu(self.conv5(x))))
        print(x.size())
        x = self.dropout(self.bn2(F.relu(self.conv6(x))))
        print(x.size())
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y=torch.nn.functional.log_softmax(x)

        
        return y

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            #print(X.type(),X.size())
            y=y.view(1,-1)[0]
            y=y.type(torch.LongTensor)
            y = y.to(device)
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                
                #print(net(X.to(device)).argmax(dim=1))
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: 
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            #print(X.type(),X.size())
            y=y.view(1,-1)[0]
            y=y.type(torch.LongTensor)
            y = y.to(device)
            #print(y.type(),y.size())
            y_hat = net(X)
            
            #print(y_hat.type(),y_hat.size())
            l = loss(y_hat, y)
            optimizer.zero_grad()
            #print('a',net.conv1.weight.grad)
            l.backward()
            
            #print('b',net.conv1.weight.grad)
            
            optimizer.step()
            
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        
if __name__ == '__main__':
    batch_size=128
    print('loading data...')
    train_iter, val_iter, test_iter = load_rotated_mnist(batch_size)
    net=PDO_eConvs()

    net.apply(init_weights)
    device = torch.device('cpu')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #net = net.cuda()

       
    lr, num_epochs = 0.001, 10
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)