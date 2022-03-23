# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:36:38 2021

@author: Rui
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
from Encoder_decoder_NN import ConvNet_3d_hard_elu_10_220_60

def post_process_1phase(label1,label2,pp_test,\
             pw_test,qw_test,tk_test,p_boun1_0,p_boun2_0,n_well,n_logk_test,nt,nz,ny,nx,\
            dx,dy,dz,bigt,x,y,t,k_test,poro,z_ref,Q,BHP,var,mean_logk,t_nday,p_ref,bo0,co,\
                vo,rou_o,gz,xp_set,yp_set,zp_set_up,zp_set_down,rw,r0,c,p0,L_t,gpu,well_image0,Phi_w):
    
    Net = ConvNet_3d_hard_elu_10_220_60(p0,L_t,label1,100,1,False)
    Net.load_state_dict(torch.load('model.ckpt'))
    Net=Net.cuda(gpu)
    Net.eval()
    
    foldername=label2+' Results'
    
    if (os.path.isdir(foldername)==False):
        os.mkdir(foldername)
    os.chdir(foldername)
    
    p_error_l2_set=np.empty((n_logk_test))
    p_R2_set=np.empty((n_logk_test))

    pp_test_pred=np.zeros((n_logk_test,nt,nz,ny,nx))
    pw_test_pred=np.zeros((n_logk_test,nt,n_well))
    qw_test_pred=np.zeros((n_logk_test,nt,n_well))

    for i in range(n_logk_test):
        p_test_pred=Net(tk_test[i]).cpu().detach().numpy()
        phi_test_pred=p_test_pred*(p_boun1_0-p_boun2_0)+p_boun2_0
        for j in range(nz):
            p_test_pred[:,:,j]=phi_test_pred[:,:,j]+rou_o*gz*(z_ref+dz*j)/1e5
        
        ki_image=torch.exp(tk_test[i,:,1:2,:,:,:]).cpu().detach().numpy()
        j_i=(2*np.pi*dz/vo/np.log(r0/rw)*c*ki_image)

        if (label1=='constq'):
            Q_image=np.zeros(ki_image.shape)
            for i_w in range(n_well):
            
                Q_image[:,:,zp_set_up[i_w]:zp_set_down[i_w]+1,yp_set[i_w],xp_set[i_w]]=\
                    ki_image[:,:,zp_set_up[i_w]:zp_set_down[i_w]+1,yp_set[i_w],xp_set[i_w]]/\
                        ki_image[:,:,zp_set_up[i_w]:zp_set_down[i_w]+1,yp_set[i_w],xp_set[i_w]].sum(axis=-1).reshape(-1,1,1)
                        
            Q_image=Q_image*Q
            pw_mat=p_test_pred-Q_image/j_i*bo0
        
            pw_col=np.zeros((nt,n_well))
            for i_well in range(n_well):
                pw_col[:,i_well]=pw_mat[:,0,zp_set_up[i_well],yp_set[i_well],xp_set[i_well]]
                
            pw_test_pred[i]=pw_col
            
        if (label1=='constp'):
            Qw_mat=(phi_test_pred-Phi_w)*j_i/bo0*well_image0
    
            Qw_col=np.zeros((nt,n_well))
            for i_well in range(n_well):
                Qw_col[:,i_well]=Qw_mat[:,0,zp_set_up[i_well]:zp_set_down[i_well]+1,yp_set[i_well],xp_set[i_well]].sum(axis=-1)
                
            qw_test_pred[i]=Qw_col
           
        p_test_pred=p_test_pred.reshape(nt,nz,ny,nx)
        
        pp_test_pred[i]=p_test_pred
    
        print('Results for '+label2+'\n')
        print('Realization %d' % (i))
        
        p_error_l2 = np.linalg.norm(pp_test[i].flatten()-pp_test_pred[i].flatten(),2)/np.linalg.norm(pp_test[i].flatten(),2)
        print('P Error L2: %e' % (p_error_l2))
        p_error_l2_set[i]=p_error_l2
        
        p_R2=1-np.sum((pp_test[i].flatten()-pp_test_pred[i].flatten())**2)/np.sum((pp_test[i].flatten()-pp_test[i].flatten().mean())**2)
        print('P coefficient of determination  R2: %e' % (p_R2))
        p_R2_set[i]=p_R2

    p_L2_mean=np.mean(p_error_l2_set)
    p_L2_var=np.var(p_error_l2_set)
    p_R2_mean=np.mean(p_R2_set)
    p_R2_var=np.var(p_R2_set)
    
    print('p L2 mean:')
    print(p_L2_mean)
    print('p L2 var:')
    print(p_L2_var)
    
    print('p R2 mean:')
    print(p_R2_mean)
    print('p R2 var:')
    print(p_R2_var)
    
    
    np.savetxt('p_'+label2+'_L2_error.txt',p_error_l2_set)
    np.savetxt('p_'+label2+'_R2.txt',p_R2_set)
    np.save('pred_pp_'+label2+'.npy',pp_test_pred)
    if (label1=='constq'):
        np.save('pred_pw_'+label1+'.npy',pw_test_pred)
    if (label1=='constp'):
        np.save('pred_qw_'+label1+'.npy',qw_test_pred)
    
    
    f=open('error'+label2+'.txt','w')
    f.write('mean p error L2:\n')
    f.write(str(p_L2_mean)+'\n')
    f.write('var p error L2:\n')
    f.write(str(p_L2_var)+'\n')
    f.write('mean p R2:\n')
    f.write(str(p_R2_mean)+'\n')
    f.write('var p R2:\n')
    f.write(str(p_R2_var)+'\n')
    f.close()
    
    #######################################################
    #结果展示    
    
    n_logk_test_plot=5

    for i_sam in range(n_logk_test_plot):

        sam1=i_sam 

        if (label1=='constq'):
            
            for i_well in range(n_well):
                plt.figure()
                plt.plot(t*bigt*t_nday,pw_test[sam1,:,i_well],'k-',label='Reference')
                plt.plot(t*bigt*t_nday,pw_test_pred[sam1,:,i_well],'r--',label='Prediction')
                plt.xlabel('Time, day',fontsize=15)
                plt.ylabel('Bottom hole pressure, bar',fontsize=15)
                plt.title("Well %d"%(i_well+1),fontsize=15)
                plt.tick_params(labelsize=13)
                plt.legend(fontsize=13)
                plt.savefig('bhp_compare_perm_%d_well_%d.png'%(sam1+1,i_well+1),bbox_inches='tight')

        if (label1=='constp'):

            for i_well in range(n_well):
                plt.figure()
                plt.plot(t*bigt*t_nday,qw_test[sam1,:,i_well],'k-',label='Reference')
                plt.plot(t*bigt*t_nday,qw_test_pred[sam1,:,i_well],'r--',label='Prediction')
                plt.xlabel('Time, day',fontsize=15)
                plt.ylabel('Flow rate, STD m$^3$/D',fontsize=15)
                plt.title("Well %d"%(i_well+1),fontsize=15)
                plt.tick_params(labelsize=13)
                plt.legend(fontsize=13)
                plt.savefig('qw_compare_perm_%d_well_%d.png'%(sam1+1,i_well+1),bbox_inches='tight')

    if (label1=='constq'):
        obs_t=np.arange(nt)
        marker_shape=['o','s','^','*','d']
        marker_color=['b','r','c','y','g']
        plt.figure(figsize=(5,5))
        plt.plot([pw_test.min()-10,pw_test.max()+10],[pw_test.min()-10,pw_test.max()+10],'k-',linewidth=2)
        for i_well in range(n_well):
            plt.scatter(pw_test[:,obs_t,i_well].flatten(),pw_test_pred[:,obs_t,i_well].flatten(),\
                    marker=marker_shape[i_well],c='',edgecolors=marker_color[i_well],label='Well %d'%(i_well+1))
              
        plt.xlabel('Reference (bar)',fontsize=18)
        plt.ylabel('Prediction (bar)',fontsize=18)
        plt.legend(fontsize=12)
        plt.savefig('bottomhole_p_compare.png',bbox_inches='tight')

    if (label1=='constp'):
        
        obs_t=np.arange(1,nt)
        marker_shape=['o','s','^','*','d']
        marker_color=['b','r','c','y','g']

        plt.figure(figsize=(5,5))
        plt.plot([qw_test.min()-50,qw_test[:,1:].max()+50],[qw_test.min()-50,qw_test[:,1:].max()+50],'k-',linewidth=2)
        for i_well in range(n_well):
            plt.scatter(qw_test[:,obs_t,i_well].flatten(),qw_test_pred[:,obs_t,i_well].flatten(),\
                    marker=marker_shape[i_well],c='',edgecolors=marker_color[i_well],label='Well %d'%(i_well+1))
        
        plt.xlabel('Reference (STD m$^3$/D)',fontsize=18)
        plt.ylabel('Prediction (STD m$^3$/D)',fontsize=18)
        plt.legend(fontsize=12)
        plt.savefig('qw_all_compare.png',bbox_inches='tight')

    ######################################################################
    ############################# error histogram #############################
    ######################################################################
    num_bins2 = 15
    plt.figure(figsize=(6,4))
    plt.hist(p_R2_set, num_bins2)
    plt.title(r'$Histogram\ \ of\ P\  R^2\ \ score$')
    plt.savefig('Histogram_P_R2.png',bbox_inches='tight')
    
    
    
    
    
    
