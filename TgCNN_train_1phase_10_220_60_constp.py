# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:33:07 2021

@author: Rui
"""
import os
import os.path
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
from statistics import mean
import sys

from other_func import tensor_time_diff2,cut_boun,cut_boun2,diff_x,diff_y,diff_z
from other_func import harmonic_mean_x,harmonic_mean_y,harmonic_mean_z,time_k_image2
from Encoder_decoder_NN import ConvNet_3d_hard_elu_10_220_60
from post_process_1phase_multi_segment_well_new import post_process_1phase

def main():
    parser = argparse.ArgumentParser()
    #### parameter input ####
    parser.add_argument('-n', '--nnodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-epoch','--num_epoch', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-lam1','--lam1',default=3,type=float,
                        help='weight of the data mismatch term in the loss function')
    parser.add_argument('-lam2','--lam2',default=0.03,type=float,
                        help='weight of the governing equation residual term in the loss function')
    parser.add_argument('-lam4','--lam4',default=0.03,type=float,
                        help='weight of the face BC residual term in the loss function')
    parser.add_argument('-lam5','--lam5',default=0.03,type=float,
                        help='weight of the corner BC residual term in the loss function')
    parser.add_argument('-lam6','--lam6',default=0.03,type=float,
                        help='weight of the line BC residual term in the loss function')
    parser.add_argument('-lr','--LR',default=0.0005,type=float,
                        help='learning rate')
    parser.add_argument('-n_decay','--n_decay',default=60,type=int,
                        help='number of epochs for performing each weight decay')
    parser.add_argument('-loss_tol','--loss_tol',default=0.0001,type=float,
                        help='training loss threshold below which the training terminates')
    parser.add_argument('-decay_rate','--decay_rate',default=0.9,type=float,
                        help='decay rate of the learning rate')
    parser.add_argument('-Batch', '--BATCH_SIZE2',default=40,type=int,
                        help='number of virtual permeability realizations in a batch')
    parser.add_argument('--local_rank',type=int, default=-1, metavar='N', help='Local process rank')
    parser.add_argument('-n_logk', '--n_logk',default=100,type=int,
                        help='number of training permeability fields with labeled pressure data')
    parser.add_argument('-n_logk_v', '--n_logk_v',default=200,type=int,
                        help='number of virtual permeabilty realizations')
    parser.add_argument('-n_logk_test', '--n_logk_test',default=50,type=int,
                        help='number of testing permeability fields used for performance testing of the trained model')
    
    
    args = parser.parse_args()
    args.world_size = args.gpus * args.nnodes
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1422'
    
    train_1phase(args)
    
def train_1phase(args):
    #### model training ####
    dist.init_process_group(backend='nccl', init_method='env://', world_size= args.world_size, rank=args.local_rank) #parallel enviroment setting
    torch.manual_seed(0)
    
    plt.rcParams['savefig.dpi'] = 300 #resolution of the plotted figures
    plt.rcParams['figure.dpi'] = 300 #resolution of the plotted figures
    label1='constp'   #wells producing at constant BHP
    path='/userhome/TgCNN_3D/3D_10_220_60_constp/'    #path to save or load data, defined by users

    ###########################################################
    ##### statistical parameters of the permeabilty fields ####
    ###########################################################
    mean_logk=4. # mean value of logrithm permeability
    var=1.       # variance of logrithm permeability
    eta=50.      # correlation length of permeablity field
    weight=0.6   # energy conservation fraction for KLE

    seed_n=38    # random seed definition
    seed=200
    seed_v=50

    ##########################################################
    ########### formation geometric properties ###############
    ##########################################################
    nx=60    # number of grids in x direction
    ny=220
    nz=10
    nt=20    # number of timesteps
    
    dx=6.096       # grid resolution in x direction, in meters
    dy=3.048
    dz=0.6096*8.5
    dt=1           
    
    L_x= nx*dx    # physical length of the reservoir in x direction, in meters
    L_y= ny*dy
    L_z= nz*dz
    
    domain=L_x*L_y*L_z
    
    x=np.arange(1,nx+1,1)
    x=x*dx
    y=np.arange(1,ny+1,1)
    y=y*dy
    z=np.arange(1,nz+1,1)
    z=z*dz
    
    L_t=nt     # number of timesteps
    t=np.linspace(1,L_t,nt)
    
    t_nday=1  # length of a timestep, in days
    bigt=1    # scale ratio of time
    t=t/bigt
    dt=dt/bigt
    L_t=L_t/bigt

    ############################################
    ####### oil and formation properties #######
    ############################################
    co=1e-4         # oil compressibility, 1/bar
    bo0=1.02        # oil formation factor
    rou_o=849/bo0   # oil underground density
    vo=3.           # oil viscosity
    cvis=0.         # oil viscosity variation factor
    crock=0.        # rock compressibility
    poro=0.2        # formation porosity, assuming uniform porosity
    gz=9.8          # gravitational acceleration
    p_ref=413.69    # reference pressure at the formation top, in bar
    z_ref=3657.6    # reference depth at the formation top, in meters

    ###################################################
    ################## well info ######################
    ###################################################
    n_well=4   # total number of production wells
       
    xp1=10           # x coordinate of well 1, in grid number
    yp1=20           # y coordinate of well 1, in grid number
    zp1_up=0         # upper z coordinate of well 1, in grid number
    zp1_down=nz-1    # lower z coordinate of well 1, in grid number
    
    xp2=50
    yp2=20
    zp2_up=0
    zp2_down=nz-1
    
    xp3=10
    yp3=200
    zp3_up=0
    zp3_down=nz-1
    
    xp4=50
    yp4=200
    zp4_up=0
    zp4_down=nz-1
    
    xp_set=[xp1,xp2,xp3,xp4]
    yp_set=[yp1,yp2,yp3,yp4]
    zp_set_up=[zp1_up,zp2_up,zp3_up,zp4_up]
    zp_set_down=[zp1_down,zp2_down,zp3_down,zp4_down]
    
    Q=50       # well flow rate, in m3/D
    BHP=350    # well BHP, in bar
    Phi_w=BHP-rou_o*gz*z_ref/1e5  # well potential value

    c=0.008527  # productivity conversion factor
    r0=0.14*np.sqrt(dx**2+dy**2)  # drainage radius
    rw=0.1                        # wellbore radius
    
    ###########################################################
    ################ obtain training data #####################
    ########################################################### 
    
    os.chdir(path)
    logk=np.load('train_logk_n=%d_mean=%d_var=%.2f_eta=%.2f.npy'%(args.n_logk,mean_logk,var,eta))
    k=np.exp(logk)
    logk_image=logk.reshape(args.n_logk,1,nz,ny,nx)
    
    bpr_array=np.load('p_train_N=%d_weight=%.2f_seed=%d_BHP=%d_var=%.2f_mean=%d_eta=%.2f_tnday=%.2f.npy'%(args.n_logk,weight,seed_n,BHP,var,mean_logk,eta,t_nday))
    Q_array=np.load('Q_train_N=%d_weight=%.2f_seed=%d_BHP=%d_var=%.2f_mean=%d_eta=%.2f_tnday=%.2f.npy'%(args.n_logk,weight,seed_n,BHP,var,mean_logk,eta,t_nday))

    pp_=bpr_array[:,1:,:,:,:].reshape(args.n_logk,nt,1,nz,ny,nx)
    qw_=Q_array[:,1:,:]
    
    p_train=pp_.copy()  
    for i in range(nz):
        p_train[:,:,:,i]=p_train[:,:,:,i]-rou_o*gz*(z_ref+dz*i)/1e5   # convert pressure to potential value
    
    tk_image= time_k_image2(t,logk_image,nx,ny,nz,args.n_logk)  # obtain input dataset to the neural network

    p_train=p_train.reshape(-1,1,nz,ny,nx)
    tk_train=tk_image.reshape(-1,2,nz,ny,nx) 
    # tkp_col=np.concatenate((tk_all_col,p_all_col),1)  
    
    # np.random.shuffle(tkp_col)
    # p_all_col=tkp_col[:,2:3]
    # tk_all_col=tkp_col[:,0:2]

    p_boun1_0=p_ref-rou_o*gz*z_ref/1e5   # upper bound of potential
    p_boun2_0=BHP-rou_o*gz*z_ref/1e5     # lower bound of potential
    
    p_train=((p_train-p_boun2_0)/(p_boun1_0-p_boun2_0))  # normalize the training pressure data based on min-max values
    p_train= torch.from_numpy(p_train)
    p_train= p_train.type(torch.FloatTensor)
    
    p0=1.
    
    tk_train= torch.from_numpy(tk_train)
    tk_train= tk_train.type(torch.FloatTensor)
    
    if args.local_rank==0:
        print('Training data collected.\n')
    
    ###################################################################
    #########  virtual permeability data extraction  ##################
    ###################################################################

    logk_v=np.load('virtual_logk_n=%d_mean=%d_var=%.2f_eta=%.2f.npy'%(args.n_logk_v,mean_logk,var,eta))
    logk_v_image=logk_v.reshape(args.n_logk_v,1,nz,ny,nx)
    
    tk_image_v= time_k_image2(t,logk_v_image,nx,ny,nz,args.n_logk_v)   # generate virtual permeability inputs to the neural network
    tk_image_v_col=tk_image_v.reshape(-1,2,nz,ny,nx) 
    
    # tk_image_v_col=tk_image_v_col.copy()
    # tk_image_v_col=np.concatenate((tk_image_v_col,tk_all_col.copy()),0)
    np.random.shuffle(tk_image_v_col)
    
    tk_train_v=torch.from_numpy(tk_image_v_col)
    tk_train_v= tk_train_v.type(torch.FloatTensor)
    
    tk_train_v_=tk_image_v_col.copy()
    tk_train_v_[:,0,:,:,:]=tk_train_v_[:,0,:,:,:]-dt
    tk_train_v_= torch.from_numpy(tk_train_v_)
    tk_train_v_= tk_train_v_.type(torch.FloatTensor)
    
    n_col=len(tk_train_v)   # number of virtual inputs
    n_train=len(p_train)    # number of labelled training inputs
    
    N_batch=math.ceil(n_col/args.BATCH_SIZE2)   # number of batches
    BATCH_SIZE=math.ceil(n_train/N_batch)       # number of training permeability realizations in a batch
    
    
    # ##################################################
    well_image0=np.zeros((1,1,nz,ny,nx))
    # define the well image binary matrix, where the grids containing the well have the value of 1, otherwise 0
    well_image0[0,0,zp1_up:zp1_down+1,yp1,xp1]=1
    well_image0[0,0,zp2_up:zp2_down+1,yp2,xp2]=1
    well_image0[0,0,zp3_up:zp3_down+1,yp3,xp3]=1
    well_image0[0,0,zp4_up:zp4_down+1,yp4,xp4]=1

    well_image0 = torch.from_numpy(well_image0)
    well_image0 = well_image0.type(torch.FloatTensor)
    well_image = cut_boun(well_image0,1)
    
    model = ConvNet_3d_hard_elu_10_220_60(p0,L_t,label1,100,1,False)   # Encoder-decoder model initialization
    torch.cuda.set_device(args.local_rank)
    model.cuda()

    well_image0=well_image0.cuda()
    well_image=well_image.cuda()

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)  # wrap the model for parallelization
    optimizer=torch.optim.Adam(model.parameters(),args.LR)   # optimization method choice

    # definition of unit conversion factors 
    fac1=0.001*8.64/bo0/vo*bigt/co
    fac3=1/t_nday*poro/bo0

    f1_train_dataset = Data.TensorDataset(tk_train,p_train)
    f2_train_dataset = Data.TensorDataset(tk_train_v,tk_train_v_)
    
    train_sampler1 = torch.utils.data.distributed.DistributedSampler(f1_train_dataset)
    
    train_sampler2 = torch.utils.data.distributed.DistributedSampler(f2_train_dataset)
    
    # two separate data loaders for training and virtual permeability realizations are used
    train_loader1 = torch.utils.data.DataLoader(dataset=f1_train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler1)
    
    train_loader2 = torch.utils.data.DataLoader(dataset=f2_train_dataset,
                                               batch_size=args.BATCH_SIZE2,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler2)

    start = datetime.now()
    total_step = len(train_loader1)
    
    lr_list=[]
    loss_set=[]
    f1_set=[]
    f2_set=[]  
    f4_set=[]  
    f5_set=[] 
    f6_set=[]
    
    for epoch in range(args.num_epoch):
        for i, data in enumerate(zip(train_loader1,train_loader2)):
            
            batch_x,batch_y=data[0]
            batch_x_v,batch_x_v_=data[1]
            batch_x=batch_x.cuda(non_blocking=True)
            batch_y=batch_y.cuda(non_blocking=True)
            batch_x_v=batch_x_v.cuda(non_blocking=True)
            batch_x_v_=batch_x_v_.cuda(non_blocking=True)
            
            pred_f=model(batch_x)
            pred_f_v=model(batch_x_v)
            pred_f_v_=model(batch_x_v_)
            
            pt=tensor_time_diff2(pred_f_v,pred_f_v_,dt) # calculate time derivatives
            pt_image=cut_boun(pt,1) # obtain internal grids other than boundaries
            
            px_l,px_r=diff_x(pred_f_v,dx)
            kx_l,kx_r=harmonic_mean_x(torch.exp(batch_x_v[:,1:2,:,:,:]))
            
            xkpx=((kx_r[:,:,1:-1,1:-1,:]*px_r[:,:,1:-1,1:-1,:]-kx_l[:,:,1:-1,1:-1,:]*px_l[:,:,1:-1,1:-1,:])/dx)
            
            py_l,py_r=diff_y(pred_f_v,dy)
            ky_l,ky_r=harmonic_mean_y(torch.exp(batch_x_v[:,1:2,:,:,:]))
            
            ykpy=((ky_r[:,:,1:-1,:,1:-1]*py_r[:,:,1:-1,:,1:-1]-ky_l[:,:,1:-1,:,1:-1]*py_l[:,:,1:-1,:,1:-1])/dy)
            
            pz_l,pz_r=diff_z(pred_f_v,dz)
            kz_l,kz_r=harmonic_mean_z(torch.exp(batch_x_v[:,1:2,:,:,:]))
            
            zkpz=((kz_r[:,:,:,1:-1,1:-1]*pz_r[:,:,:,1:-1,1:-1]-kz_l[:,:,:,1:-1,1:-1]*pz_l[:,:,:,1:-1,1:-1])/dz)
            
            f1=torch.pow((pred_f-batch_y)*args.lam1,2).mean()  # calculation of data mismatch term in the loss function
            
            j_in=(2*np.pi*dz/vo/np.log(r0/rw)*c*torch.exp(batch_x_v[:,1:2,:,:,:]))  # productivity index
            Q=(pred_f_v*(p_boun1_0-p_boun2_0)+p_boun2_0-Phi_w)*j_in/bo0   # well flow rate calculation
            
            fac2=Q/dx/dy/dz*bigt/co/(p_boun1_0-p_boun2_0)  # conversion factor
            
            fac2_internal=cut_boun(fac2,1)
            residual=fac3*pt_image-fac1*xkpx-fac1*ykpy-fac1*zkpz+fac2_internal*well_image
            f2=torch.pow((residual)*args.lam2,2).mean()  # calculation of the governing equation residuals in internal grids of the reservoir
            
            pxy_nof_byu=-(ky_r[:,:,1:-1,-1:,1:-1]*py_r[:,:,1:-1,-1:,1:-1])/dy+\
                (kx_r[:,:,1:-1,-1:,:]*px_r[:,:,1:-1,-1:,:]-kx_l[:,:,1:-1,-1:,:]*px_l[:,:,1:-1,-1:,:])/dx+\
                (kz_r[:,:,:,-1:,1:-1]*pz_r[:,:,:,-1:,1:-1]-kz_l[:,:,:,-1:,1:-1]*pz_l[:,:,:,-1:,1:-1])/dz
            pxy_nof_byd=(ky_l[:,:,1:-1,0:1,1:-1]*py_l[:,:,1:-1,0:1,1:-1])/dy+\
                (kx_r[:,:,1:-1,0:1,:]*px_r[:,:,1:-1,0:1,:]-kx_l[:,:,1:-1,0:1,:]*px_l[:,:,1:-1,0:1,:])/dx+\
                (kz_r[:,:,:,0:1,1:-1]*pz_r[:,:,:,0:1,1:-1]-kz_l[:,:,:,0:1,1:-1]*pz_l[:,:,:,0:1,1:-1])/dz
            pxy_nof_bxr=-(kx_r[:,:,1:-1,1:-1,-1:]*px_r[:,:,1:-1,1:-1,-1:])/dx+\
                (ky_r[:,:,1:-1,:,-1:]*py_r[:,:,1:-1,:,-1:]-ky_l[:,:,1:-1,:,-1:]*py_l[:,:,1:-1,:,-1:])/dy+\
                (kz_r[:,:,:,1:-1,-1:]*pz_r[:,:,:,1:-1,-1:]-kz_l[:,:,:,1:-1,-1:]*pz_l[:,:,:,1:-1,-1:])/dz
            pxy_nof_bxl=(kx_l[:,:,1:-1,1:-1,0:1]*px_l[:,:,1:-1,1:-1,0:1])/dx+\
                (ky_r[:,:,1:-1,:,0:1]*py_r[:,:,1:-1,:,0:1]-ky_l[:,:,1:-1,:,0:1]*py_l[:,:,1:-1,:,0:1])/dy+\
                (kz_r[:,:,:,1:-1,0:1]*pz_r[:,:,:,1:-1,0:1]-kz_l[:,:,:,1:-1,0:1]*pz_l[:,:,:,1:-1,0:1])/dz    
            pxy_nof_bzr=-(kz_r[:,:,-1:,1:-1,1:-1]*pz_r[:,:,-1:,1:-1,1:-1])/dz+\
                (ky_r[:,:,-1:,:,1:-1]*py_r[:,:,-1:,:,1:-1]-ky_l[:,:,-1:,:,1:-1]*py_l[:,:,-1:,:,1:-1])/dy+\
                (kx_r[:,:,-1:,1:-1,:]*px_r[:,:,-1:,1:-1,:]-kx_l[:,:,-1:,1:-1,:]*px_l[:,:,-1:,1:-1,:])/dx
            pxy_nof_bzl=(kz_l[:,:,0:1,1:-1,1:-1]*pz_l[:,:,0:1,1:-1,1:-1])/dz+\
                (ky_r[:,:,0:1,:,1:-1]*py_r[:,:,0:1,:,1:-1]-ky_l[:,:,0:1,:,1:-1]*py_l[:,:,0:1,:,1:-1])/dy+\
                (kx_r[:,:,0:1,1:-1,:]*px_r[:,:,0:1,1:-1,:]-kx_l[:,:,0:1,1:-1,:]*px_l[:,:,0:1,1:-1,:])/dx
                
            no_flow_residual_y1=fac3*pt[:,:,1:-1,-1:,1:-1]-fac1*pxy_nof_byu
            no_flow_residual_y2=fac3*pt[:,:,1:-1,0:1,1:-1]-fac1*pxy_nof_byd
            no_flow_residual_x1=fac3*pt[:,:,1:-1,1:-1,-1:]-fac1*pxy_nof_bxr
            no_flow_residual_x2=fac3*pt[:,:,1:-1,1:-1,0:1]-fac1*pxy_nof_bxl
            no_flow_residual_z1=fac3*pt[:,:,-1:,1:-1,1:-1]-fac1*pxy_nof_bzr+cut_boun2(fac2[:,:,0:1],1)*cut_boun2(well_image0[:,:,0:1],1)
            no_flow_residual_z2=fac3*pt[:,:,0:1,1:-1,1:-1]-fac1*pxy_nof_bzl+cut_boun2(fac2[:,:,-1:],1)*cut_boun2(well_image0[:,:,-1:],1)
            
            f4=torch.pow((no_flow_residual_x1)*args.lam4,2).mean()+\
                torch.pow((no_flow_residual_x2)*args.lam4,2).mean()+\
                torch.pow((no_flow_residual_y1)*args.lam4,2).mean()+\
                torch.pow((no_flow_residual_y2)*args.lam4,2).mean()+\
                torch.pow((no_flow_residual_z1)*args.lam4,2).mean()+\
                torch.pow((no_flow_residual_z2)*args.lam4,2).mean()      # calculation of the governing equation residuals at no-flow boundaries (six faces) of the reservoir
                
                
            cor000=(kx_l[:,:,0,0,0]*px_l[:,:,0,0,0])/dx+\
                (ky_l[:,:,0,0,0]*py_l[:,:,0,0,0])/dy+\
                    (kz_l[:,:,0,0,0]*pz_l[:,:,0,0,0])/dz
                    
            cor001=(-kx_r[:,:,0,0,-1]*px_r[:,:,0,0,-1])/dx+\
                (ky_l[:,:,0,0,-1]*py_l[:,:,0,0,-1])/dy+\
                    (kz_l[:,:,0,0,-1]*pz_l[:,:,0,0,-1])/dz
                    
            cor010=(kx_l[:,:,0,-1,0]*px_l[:,:,0,-1,0])/dx+\
                (-ky_r[:,:,0,-1,0]*py_r[:,:,0,-1,0])/dy+\
                    (kz_l[:,:,0,-1,0]*pz_l[:,:,0,-1,0])/dz
                    
            cor011=(-kx_r[:,:,0,-1,-1]*px_r[:,:,0,-1,-1])/dx+\
                (-ky_r[:,:,0,-1,-1]*py_r[:,:,0,-1,-1])/dy+\
                    (kz_l[:,:,0,-1,-1]*pz_l[:,:,0,-1,-1])/dz
                    
            cor100=(kx_l[:,:,-1,0,0]*px_l[:,:,-1,0,0])/dx+\
                (ky_l[:,:,-1,0,0]*py_l[:,:,-1,0,0])/dy+\
                    (-kz_r[:,:,-1,0,0]*pz_r[:,:,-1,0,0])/dz
                    
            cor101=(-kx_r[:,:,-1,0,-1]*px_r[:,:,-1,0,-1])/dx+\
                (ky_l[:,:,-1,0,-1]*py_l[:,:,-1,0,-1])/dy+\
                    (-kz_r[:,:,-1,0,-1]*pz_r[:,:,-1,0,-1])/dz
                    
            cor110=(kx_l[:,:,-1,-1,0]*px_l[:,:,-1,-1,0])/dx+\
                (-ky_r[:,:,-1,-1,0]*py_r[:,:,-1,-1,0])/dy+\
                    (-kz_r[:,:,-1,-1,0]*pz_r[:,:,-1,-1,0])/dz
                    
            cor111=(-kx_r[:,:,-1,-1,-1]*px_r[:,:,-1,-1,-1])/dx+\
                (-ky_r[:,:,-1,-1,-1]*py_r[:,:,-1,-1,-1])/dy+\
                    (-kz_r[:,:,-1,-1,-1]*pz_r[:,:,-1,-1,-1])/dz
    
            cor_res000=fac3*pt[:,:,0,0,0]-fac1*cor000
            cor_res001=fac3*pt[:,:,0,0,-1]-fac1*cor001
            cor_res010=fac3*pt[:,:,0,-1,0]-fac1*cor010
            cor_res011=fac3*pt[:,:,0,-1,-1]-fac1*cor011
            cor_res100=fac3*pt[:,:,-1,0,0]-fac1*cor100
            cor_res101=fac3*pt[:,:,-1,0,-1]-fac1*cor101
            cor_res110=fac3*pt[:,:,-1,-1,0]-fac1*cor110
            cor_res111=fac3*pt[:,:,-1,-1,-1]-fac1*cor111
            
                 
            f5=torch.pow((cor_res000)*args.lam5,2).mean()+\
                torch.pow((cor_res001)*args.lam5,2).mean()+\
                torch.pow((cor_res010)*args.lam5,2).mean()+\
                torch.pow((cor_res011)*args.lam5,2).mean()+\
                torch.pow((cor_res100)*args.lam5,2).mean()+\
                torch.pow((cor_res101)*args.lam5,2).mean()+\
                torch.pow((cor_res110)*args.lam5,2).mean()+\
                torch.pow((cor_res111)*args.lam5,2).mean()    # calculation of the governing equation residuals at the eight corners of the reservoir
            
           
            line_00x= (kx_r[:,:,0,0,:]*px_r[:,:,0,0,:]-kx_l[:,:,0,0,:]*px_l[:,:,0,0,:])/dx+\
                          (ky_l[:,:,0,0,1:-1]*py_l[:,:,0,0,1:-1])/dy+\
                          (kz_l[:,:,0,0,1:-1]*pz_l[:,:,0,0,1:-1])/dz
                         
            line_01x= (kx_r[:,:,0,-1,:]*px_r[:,:,0,-1,:]-kx_l[:,:,0,-1,:]*px_l[:,:,0,-1,:])/dx+\
                          (-ky_r[:,:,0,-1,1:-1]*py_r[:,:,0,-1,1:-1])/dy+\
                          (kz_l[:,:,0,-1,1:-1]*pz_l[:,:,0,0,1:-1])/dz
                         
            line_0y0= (ky_r[:,:,0,:,0]*py_r[:,:,0,:,0]-ky_l[:,:,0,:,0]*py_l[:,:,0,:,0])/dy+\
                          (kx_l[:,:,0,1:-1,0]*px_l[:,:,0,1:-1,0])/dx+\
                          (kz_l[:,:,0,1:-1,0]*pz_l[:,:,0,1:-1,0])/dz
                         
            line_0y1= (ky_r[:,:,0,:,-1]*py_r[:,:,0,:,-1]-ky_l[:,:,0,:,-1]*py_l[:,:,0,:,-1])/dy+\
                          (-kx_r[:,:,0,1:-1,-1]*px_r[:,:,0,1:-1,-1])/dx+\
                          (kz_l[:,:,0,1:-1,-1]*pz_l[:,:,0,1:-1,-1])/dz
                         
            line_z00= (kz_r[:,:,:,0,0]*pz_r[:,:,:,0,0]-kz_l[:,:,:,0,0]*pz_l[:,:,:,0,0])/dz+\
                          (kx_l[:,:,1:-1,0,0]*px_l[:,:,1:-1,0,0])/dx+\
                          (ky_l[:,:,1:-1,0,0]*py_l[:,:,1:-1,0,0])/dy
                         
            line_z01= (kz_r[:,:,:,0,-1]*pz_r[:,:,:,0,-1]-kz_l[:,:,:,0,-1]*pz_l[:,:,:,0,-1])/dz+\
                          (-kx_r[:,:,1:-1,0,-1]*px_r[:,:,1:-1,0,-1])/dx+\
                          (ky_l[:,:,1:-1,0,-1]*py_l[:,:,1:-1,0,-1])/dy
                         
            line_z10= (kz_r[:,:,:,-1,0]*pz_r[:,:,:,-1,0]-kz_l[:,:,:,-1,0]*pz_l[:,:,:,-1,0])/dz+\
                          (kx_l[:,:,1:-1,-1,0]*px_l[:,:,1:-1,-1,0])/dx+\
                          (-ky_r[:,:,1:-1,-1,0]*py_r[:,:,1:-1,-1,0])/dy
                         
            line_z11= (kz_r[:,:,:,-1,-1]*pz_r[:,:,:,-1,-1]-kz_l[:,:,:,-1,-1]*pz_l[:,:,:,-1,-1])/dz+\
                          (-kx_r[:,:,1:-1,-1,-1]*px_r[:,:,1:-1,-1,-1])/dx+\
                          (-ky_r[:,:,1:-1,-1,-1]*py_r[:,:,1:-1,-1,-1])/dy
                         
            line_10x= (kx_r[:,:,-1,0,:]*px_r[:,:,-1,0,:]-kx_l[:,:,-1,0,:]*px_l[:,:,-1,0,:])/dx+\
                          (ky_l[:,:,-1,0,1:-1]*py_l[:,:,-1,0,1:-1])/dy+\
                          (-kz_r[:,:,-1,0,1:-1]*pz_r[:,:,-1,0,1:-1])/dz
                         
            line_11x= (kx_r[:,:,-1,-1,:]*px_r[:,:,-1,-1,:]-kx_l[:,:,-1,-1,:]*px_l[:,:,-1,-1,:])/dx+\
                          (-ky_r[:,:,-1,-1,1:-1]*py_r[:,:,-1,-1,1:-1])/dy+\
                          (-kz_r[:,:,-1,-1,1:-1]*pz_r[:,:,-1,-1,1:-1])/dz  
                         
            line_1y0= (ky_r[:,:,-1,:,0]*py_r[:,:,-1,:,0]-ky_l[:,:,-1,:,0]*py_l[:,:,-1,:,0])/dy+\
                          (kx_l[:,:,-1,1:-1,0]*px_l[:,:,-1,1:-1,0])/dx+\
                          (-kz_r[:,:,-1,1:-1,0]*pz_r[:,:,-1,1:-1,0])/dz  
                         
            line_1y1= (ky_r[:,:,-1,:,-1]*py_r[:,:,-1,:,-1]-ky_l[:,:,-1,:,-1]*py_l[:,:,-1,:,-1])/dy+\
                          (-kx_r[:,:,-1,1:-1,-1]*px_r[:,:,-1,1:-1,-1])/dx+\
                          (-kz_r[:,:,-1,1:-1,-1]*pz_r[:,:,-1,1:-1,-1])/dz  
                         
                         
            line_res00x = fac3*pt[:,:,0,0,1:-1]-fac1*line_00x
            line_res01x = fac3*pt[:,:,0,-1,1:-1]-fac1*line_01x
            line_res0y0 = fac3*pt[:,:,0,1:-1,0]-fac1*line_0y0
            line_res0y1 = fac3*pt[:,:,0,1:-1,-1]-fac1*line_0y1
            line_resz00 = fac3*pt[:,:,1:-1,0,0]-fac1*line_z00
            line_resz01 = fac3*pt[:,:,1:-1,0,-1]-fac1*line_z01
            line_resz10 = fac3*pt[:,:,1:-1,-1,0]-fac1*line_z10
            line_resz11 = fac3*pt[:,:,1:-1,-1,-1]-fac1*line_z11
            line_res10x = fac3*pt[:,:,-1,0,1:-1]-fac1*line_10x
            line_res11x = fac3*pt[:,:,-1,-1,1:-1]-fac1*line_11x
            line_res1y0 = fac3*pt[:,:,-1,1:-1,0]-fac1*line_1y0
            line_res1y1 = fac3*pt[:,:,-1,1:-1,-1]-fac1*line_1y1
            
            f6=torch.pow((line_res00x)*args.lam6,2).mean()+\
                torch.pow((line_res01x)*args.lam6,2).mean()+\
                torch.pow((line_res0y0)*args.lam6,2).mean()+\
                torch.pow((line_res0y1)*args.lam6,2).mean()+\
                torch.pow((line_resz00)*args.lam6,2).mean()+\
                torch.pow((line_resz01)*args.lam6,2).mean()+\
                torch.pow((line_resz10)*args.lam6,2).mean()+\
                torch.pow((line_resz11)*args.lam6,2).mean()+\
                torch.pow((line_res10x)*args.lam6,2).mean()+\
                torch.pow((line_res11x)*args.lam6,2).mean()+\
                torch.pow((line_res1y0)*args.lam6,2).mean()+\
                torch.pow((line_res1y1)*args.lam6,2).mean()   # calculation of the governing equation residuals at the 12 boundary lines of the reservoir

            loss=f1+f2+f4+f5+f6

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Step [{}/{}], Rank [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epoch, i + 1, total_step, args.local_rank, args.world_size, loss))

            loss_set.append(loss.item())
            f1_set.append(f1.item())
            f2_set.append(f2.item())
            f4_set.append(f4.item())
            f5_set.append(f4.item())
            f6_set.append(f4.item())
            lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        
        dist.barrier()    
                
        if (epoch+1) % args.n_decay == 0:
            # learning rate decay
            for p in optimizer.param_groups:
                p['lr'] *= args.decay_rate

        if mean(loss_set[-10:])<args.loss_tol:
            # early termination of the training process if certain criteria is met
            print('Loss converged.\n')
            break

    if args.local_rank == 0:
        elapsed=str(datetime.now() - start)
        print("Training complete in: " + elapsed)
        
        folder=path+'meanlogk=%d_var=%.2f_eta=%.2f_lam1=%.2f_lam2=%.2f_lam4=%.2f_lam5=%.2f_lam6=%.2f_LR=%.5f_epoch=%d_DDP'%(mean_logk,var,eta,args.lam1,args.lam2,args.lam4,args.lam5,args.lam6,args.LR,epoch+1)
        if os.path.exists(folder):
            folder=folder+'_1'
        os.mkdir(folder)
        os.chdir(folder)
        
        plt.figure()     
        plt.plot(range(len(loss_set)),loss_set)
        plt.xlabel('Iteration')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.savefig('total_loss.png',bbox_inches='tight')
        
        plt.figure()   
        plt.plot(range(len(f1_set)),f1_set,label='data')
        plt.plot(range(len(f2_set)),f2_set,label='Internal')  
        plt.plot(range(len(f4_set)),f4_set,label='Face')
        plt.plot(range(len(f5_set)),f5_set,label='Corner') 
        plt.plot(range(len(f6_set)),f6_set,label='Line') 
        plt.xlabel('Iteration')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig('separate_loss.png',bbox_inches='tight')
        
        # plt.figure()     
        # plt.plot(range(len(f5_set)),f5_set)      
        # plt.xlabel('Iteration')
        # plt.ylabel('f5_loss')
        # plt.yscale('log')  
        # plt.savefig('f5_loss.png',bbox_inches='tight')  
        
        # plt.figure()     
        # plt.plot(range(len(f6_set)),f6_set)      
        # plt.xlabel('Iteration')
        # plt.ylabel('f6_loss') 
        # plt.yscale('log')  
        # plt.savefig('f6_loss.png',bbox_inches='tight') 
        
        plt.figure()  
        plt.plot(range(len(lr_list)),lr_list,color = 'r')  
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')  
        plt.savefig('LR.png',bbox_inches='tight')  

        torch.save(model.module.state_dict(),'model.ckpt')
        
        np.savetxt('loss.txt',loss_set)
        np.savetxt('data_loss.txt',f1_set)
        np.savetxt('PDE_loss.txt',f2_set)
        np.savetxt('BC_loss.txt',f4_set)
        # np.savetxt('corner_loss.txt',f5_set)
        # np.savetxt('line_loss.txt',f6_set)
        np.savetxt('Learning_rate.txt',lr_list)
        
        f=open('training parameters.txt','w')
        f.write('mean_logk: %d, var: %.2f, eta: %.2f, weight: %.2f\n'%(mean_logk,var,eta,weight)+\
                'n_logk_train: %d, n_logk_test: %d, n_logk_virtual: %d\n'%(args.n_logk,args.n_logk_test,args.n_logk_v)+\
                'nx: %d, ny: %d, nz: %d, nt: %d\n'%(nx,ny,nz,nt)+\
                'dx: %.2f, dy: %.2f, dz: %.2f, t_nday: %.2f\n'%(dx,dy,dz,t_nday)+\
                'BHP_prd: %d\n'%(BHP)+\
                'BATCH_SIZE2: %d, n_batch: %d, lr: %.5f, epoch: %d\n'%(args.BATCH_SIZE2,N_batch,args.LR,epoch+1)+\
                'lam1: %.2f, lam2: %.2f, lam4: %.2f, lam5: %.2f, lam6: %.2f,  loss_tol: %.5f, n_decay: %d, decay_rate: %.2f\n'%(args.lam1,args.lam2,args.lam4,args.lam5,args.lam6,args.loss_tol,args.n_decay,args.decay_rate)+\
                'training time: %s\n'%(elapsed)+\
                'p_boun1_0: %.2f, p_boun2_0: %.2f\n'%(p_boun1_0,p_boun2_0)+\
                'xp_set: '+str(xp_set)+'\n'+\
                'yp_set: '+str(yp_set)+'\n'+\
                'zp_set_up: '+str(zp_set_up)+'\n'+\
                'zp_set_down: '+str(zp_set_down)+'\n'
                )
        f.close()

        #########################################################################
        ####### Model evaluation on the training set ############################
        #########################################################################
        
        tk_image=torch.from_numpy(tk_image).type(torch.FloatTensor).cuda()
       
        os.chdir(folder)
        
        post_process_1phase(label1,'train',pp_.reshape(args.n_logk,nt,nz,ny,nx),\
             'dummy',qw_,tk_image,p_boun1_0,p_boun2_0,n_well,args.n_logk,nt,nz,ny,nx,\
            dx,dy,dz,bigt,x,y,t,k,poro,z_ref,Q,BHP,var,mean_logk,t_nday,p_ref,bo0,co,\
                vo,rou_o,gz,xp_set,yp_set,zp_set_up,zp_set_down,rw,r0,c,p0,L_t,args.local_rank,well_image0.cpu().numpy(),Phi_w)
        
        ##########################################################################
        ####### Model evaluation on the testing set ############################
        #########################################################################
        
        os.chdir(path)
        
        logk_test=np.load('test_logk_n=%d_mean=%d_var=%.2f_eta=%.2f.npy'%(args.n_logk_test,mean_logk,var,eta))
        
        k_test=np.exp(logk_test)
        
        logk_test_image=logk_test.reshape(args.n_logk_test,1,nz,ny,nx)

        bpr_array_test=np.load('p_test_N=%d_weight=%.2f_seed=%d_BHP=%d_var=%.2f_mean=%d_eta=%.2f_tnday=%.2f.npy'%(args.n_logk_test,weight,seed,BHP,var,mean_logk,eta,t_nday))
        Q_array_test=np.load('Q_test_N=%d_weight=%.2f_seed=%d_BHP=%d_var=%.2f_mean=%d_eta=%.2f_tnday=%.2f.npy'%(args.n_logk_test,weight,seed,BHP,var,mean_logk,eta,t_nday))

        pp_test=bpr_array_test[:,1:,:,:,:]
        qw_test=Q_array_test[:,1:,:]
        
        tk_test= time_k_image2(t,logk_test_image,nx,ny,nz,args.n_logk_test)
        tk_test = torch.from_numpy(tk_test)
        tk_test = tk_test.type(torch.FloatTensor)
        tk_test = tk_test.cuda()
             
        os.chdir(folder)
        
        post_process_1phase(label1,'test',pp_test,\
             'dummy',qw_test,tk_test,p_boun1_0,p_boun2_0,n_well,args.n_logk_test,nt,nz,ny,nx,\
            dx,dy,dz,bigt,x,y,t,k_test,poro,z_ref,Q,BHP,var,mean_logk,t_nday,p_ref,bo0,co,\
                vo,rou_o,gz,xp_set,yp_set,zp_set_up,zp_set_down,rw,r0,c,p0,L_t,args.local_rank,well_image0.cpu().numpy(),Phi_w)

    dist.barrier()
    sys.exit(0)

if __name__ == '__main__':
    main()
    