# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:31:31 2021

@author: Rui
"""

#### The class defines the architecture of the encoder-decoder network used in the training process ####

import torch
import torch.nn as nn

#### Define swish activation function ####
class Swish(nn.Module):
	def __init__(self, inplace=True):
		super(Swish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		if self.inplace:
			x.mul_(torch.sigmoid(x))
			return x
		else:
			return x * torch.sigmoid(x)

torch.set_printoptions(precision=7, threshold=None, edgeitems=None, linewidth=None, profile=None)

########## Encoder-decoder network design ###########
class ConvNet_3d_hard_elu_10_220_60(nn.Module):
    def __init__(self,p0,lt, label1, num_code=100,out_channel=1,bn=False):

        ''' p0: normalized initial pressure distribution
            lt: length of total timestep
            label1: well control case labeling, choose from 'constp' or 'constq', indicating wells producing at constant BHP or constant flow rate, respectively
            num_code: number of neurons(codes) in the hidden layer of the fully-connected section, default setting is 100
            out_channel: number of output channels, default setting is 1
            bn: whether or not to perform batch normalization, default setting is False
        ''' 

        super(ConvNet_3d_hard_elu_10_220_60, self).__init__()
        self.p0=p0
        self.lt=lt
        self.label1=label1

        #### encoder layers design ####
        if bn:
            self.encode = nn.Sequential(
                nn.Conv3d(2, 16, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm3d(16),
                Swish(),  
    
                nn.Conv3d(16, 32, kernel_size=[3,5,5], stride=[1,2,2], padding=0),
                nn.BatchNorm3d(32),
                Swish(),
    
                nn.Conv3d(32, 64, kernel_size=3, stride=[1,2,2], padding=0),
                nn.BatchNorm3d(64),
                Swish(),
                
                nn.Conv3d(64, 128, kernel_size=[3,5,5], stride=[1,2,2], padding=0),
                nn.BatchNorm3d(128),
                Swish()
                )
        else:
            self.encode = nn.Sequential(
                nn.Conv3d(2, 16, kernel_size=4, stride=1, padding=0),               
                Swish(),
        
                nn.Conv3d(16, 32, kernel_size=[3,5,5], stride=[1,2,2], padding=0),         
                Swish(),
                
                nn.Conv3d(32, 64, kernel_size=3, stride=[1,2,2], padding=0),
                Swish(),
                
                nn.Conv3d(64, 128, kernel_size=[3,5,5], stride=[1,2,2], padding=0),
                Swish()
                )    

        #### fully-connected layer design ####
        self.fc = nn.Sequential(                
        nn.Linear(128*1 *25*5, num_code),
        nn.Linear(num_code, 128*1 *25*5),
        )

        #### decoder layers design ####
        if bn:
            self.decode = nn.Sequential(
                                
                nn.ConvTranspose3d( 128, 64, [3,5,5], [1,2,2], 0),
                nn.BatchNorm3d(64),
                Swish(),
                               
                nn.ConvTranspose3d( 64, 32, 3, [1,2,2], 0),
                nn.BatchNorm3d(32),
                Swish(),
                   
                nn.ConvTranspose3d( 32, 16, [3,5,5], [1,2,2], 0),
                nn.BatchNorm3d(16),
                Swish(),
                
                nn.ConvTranspose3d( 16, out_channel, 4, 1, 0),
                nn.ELU())
    
        else:
            self.decode = nn.Sequential(

                nn.ConvTranspose3d( 128, 64, [3,5,5], [1,2,2], 0),               
                Swish(),
                             
                nn.ConvTranspose3d( 64, 32, 3, [1,2,2], 0),
                Swish(),
                
                nn.ConvTranspose3d( 32, 16, [3,5,5], [1,2,2], 0),
                Swish(),
         
                nn.ConvTranspose3d( 16, out_channel, 4, 1, 0),
                nn.ELU())           

       
    def forward(self, x):
        code = self.encode(x)
        code = code.view(code.size(0), -1)
        code = self.fc(code)
        code = code.view(-1, 128,1,25, 5)
        out = self.decode(code)
        
        #### hard constain the intial condition, when t=0, force the output to be p0 ####
        if self.label1=='constq':
            out = self.p0*(1-(x[:,0:1,:,:,:]/self.lt))-(x[:,0:1,:,:,:]/self.lt)*out
        
        return out   