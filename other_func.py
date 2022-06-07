# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:31:31 2021

@author: Rui
"""
import numpy as np

def tensor_time_diff2(h,h_,dt):
    #### calculate time derivative of pressure ####
    ht=(h-h_)/dt
    return ht

def cut_boun(u,nb):
    #### extract internal grids cutting nb boundary ones ####
    u_cut=u[:,:,nb:-nb,nb:-nb,nb:-nb]
    return u_cut

def cut_boun2(u,nb):
    u_cut=u[:,:,:,nb:-nb,nb:-nb]
    return u_cut

def diff_x(u,dx):
    #### calculate left or right spatial derivatives of pressure in x direction ####
    u_l=u[:,:,:,:,0:-2]
    u_c=u[:,:,:,:,1:-1]
    u_r=u[:,:,:,:,2:]
    
    diff_u_left=(u_c-u_l)/dx
    diff_u_right=(u_r-u_c)/dx
    return diff_u_left,diff_u_right

def diff_y(u,dy):
    #### calculate front or back spatial derivatives of pressure in y direction ####
    u_l=u[:,:,:,0:-2,:]
    u_c=u[:,:,:,1:-1,:]
    u_u=u[:,:,:,2:,:]
    
    diff_u_low=(u_c-u_l)/dy
    diff_u_up=(u_u-u_c)/dy
    return diff_u_low,diff_u_up

def diff_z(u,dz):
    #### calculate lower or upper spatial derivatives of pressure in z direction ####
    u_l=u[:,:,0:-2,:,:]
    u_c=u[:,:,1:-1,:,:]
    u_u=u[:,:,2:,:,:]
    
    diff_u_low=(u_c-u_l)/dz
    diff_u_up=(u_u-u_c)/dz
    return diff_u_low,diff_u_up

def harmonic_mean_x(k):
    #### caluclate the harmonic mean of permeability fields in x direction ####
    k_l=k[:,:,:,:,0:-2]
    k_c=k[:,:,:,:,1:-1]
    k_r=k[:,:,:,:,2:]
    
    mean_k_left=2*k_c*k_l/(k_c+k_l)
    mean_k_right=2*k_c*k_r/(k_c+k_r)   
    
    return mean_k_left,mean_k_right

def harmonic_mean_y(k):
    #### caluclate the harmonic mean of permeability fields in y direction ####
    k_l=k[:,:,:,0:-2,:]
    k_c=k[:,:,:,1:-1,:]
    k_u=k[:,:,:,2:,:]
    
    mean_k_low=2*k_c*k_l/(k_c+k_l)
    mean_k_up=2*k_c*k_u/(k_c+k_u)   
    
    return mean_k_low,mean_k_up

def harmonic_mean_z(k):
    #### caluclate the harmonic mean of permeability fields in z direction ####
    k_l=k[:,:,0:-2,:,:]
    k_c=k[:,:,1:-1,:,:]
    k_u=k[:,:,2:,:,:]
    
    mean_k_low=2*k_c*k_l/(k_c+k_l)
    mean_k_up=2*k_c*k_u/(k_c+k_u)   
    
    return mean_k_low,mean_k_up


def time_k_image2(t,k,nx,ny,nz,n_logk):
    #### define the input to the neural network, including two channels, namely a 3D time matrix and a 3D permeability field image
    nt=len(t)
    tk_image=np.zeros((n_logk,nt,2,nz,ny,nx))
    for i_k in range(n_logk):
        for i_t in range(nt):
            tk_image[i_k,i_t:i_t+1,0:1,:,:,:]=t[i_t]*np.ones((1,1,nz,ny,nx))
            tk_image[i_k,i_t:i_t+1,1:2,:,:,:]=k[i_k:i_k+1]
    return tk_image