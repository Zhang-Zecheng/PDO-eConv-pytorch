# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:57:48 2020

@author: roderickzzc
"""
from model import common
from model import mwpdo_layers
import torch
import torch.nn as nn
import scipy.io as sio


def make_model(args, parent=False):
    return MWPDO(args)

class MWPDO(nn.Module):
    def __init__(self, args):
        super(MWPDO, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.scale_idx = 0
        nColor = args.n_colors

        act = nn.ReLU(True)

       
        self.DWT = common.DWT()
        self.IWT = common.IWT()

        n = 1
        m_head = [mwpdo_layers.BBlock(nColor, n_feats, p=8, act=act,bn=False)]
        d_l0 = []
        d_l0.append(mwpdo_layers.DBlock_com1(n_feats, n_feats,p=8,act=act,bn=False))


        d_l1 = [mwpdo_layers.BBlock1(n_feats * 4, n_feats * 2, p=8,act=act,bn=False)]
        d_l1.append(mwpdo_layers.DBlock_com1(n_feats * 2, n_feats * 2, p=8, act=act,bn=False))

        d_l2 = []
        d_l2.append(mwpdo_layers.BBlock1(n_feats * 8, n_feats * 4, p=8, act=act,bn=False))
        d_l2.append(mwpdo_layers.DBlock_com1(n_feats * 4, n_feats * 4, p=8, act=act,bn=False))
        pro_l3 = []
        pro_l3.append(mwpdo_layers.BBlock1(n_feats * 16, n_feats * 8, p=8, act=act,bn=False))
        pro_l3.append(mwpdo_layers.DBlock_com(n_feats * 8, n_feats * 8, p=8, act=act,bn=False))
        pro_l3.append(mwpdo_layers.DBlock_inv(n_feats * 8, n_feats * 8, p=8, act=act,bn=False))
        pro_l3.append(mwpdo_layers.BBlock1(n_feats * 8, n_feats * 16, p=8, act=act,bn=False))

        i_l2 = [mwpdo_layers.DBlock_inv1(n_feats * 4, n_feats * 4, p=8, act=act,bn=False)]
        i_l2.append(mwpdo_layers.BBlock1(n_feats * 4, n_feats * 8, p=8, act=act,bn=False))

        i_l1 = [mwpdo_layers.DBlock_inv1(n_feats * 2, n_feats * 2, p=8, act=act,bn=False)]
        i_l1.append(mwpdo_layers.BBlock1(n_feats * 2, n_feats * 4, p=8, act=act,bn=False))

        i_l0 = [mwpdo_layers.DBlock_inv1(n_feats, n_feats, p=8, act=act,bn=False)]

        #m_tail = [mwpdo_layers.g_conv2d(n_feats, nColor, p=8,partial=mwpdo_layers.partial_dict_0,tran=mwpdo_layers.tran_to_partial_coef_0)]
        m_tail = [common.default_conv(n_feats*8, nColor, kernel_size)]
        
        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #print('x', x.size())
        x0 = self.d_l0(self.head(x))
        x1 = self.d_l1(self.DWT(x0))
        #print('x1',x1.size())
        x2 = self.d_l2(self.DWT(x1))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1
        x_ = self.IWT(self.i_l1(x_)) + x0
        x = self.tail(self.i_l0(x_)) + x
        

        return x

