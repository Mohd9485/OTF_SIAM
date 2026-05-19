#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:57:08 2024

@author: jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import odeint
import torch, math, time
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import sys
from EnKF import EnKF
from SIR import SIR
from OT_enkf import OT
# from OT import OT
from scipy.integrate import  RK45
from OT_reg import OT_reg
#%matplotlib auto

# plt.close('all')

randint = np.random.randint(0,1000)
randint = 0 #694# 748 # 101 # 704 # 980  ##### 748 for 10 
print(randint)
np.random.seed(randint)
torch.manual_seed(randint)


plt.close('all')
# Choose h(x) here, the observation rule
def h(x):
    return x[observation_idx,]
# =============================================================================
#     return H@x #np.matmul(H,x)
# =============================================================================


def L96(t, x):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector

    d = np.zeros_like(x)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(L):
        d[i] = (x[(i + 1) % L] - x[i - 2]) * x[i - 1] - x[i] + F   
    return d

def ML96(t, x):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    x = x.reshape(L,-1)
    d = np.zeros_like(x)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(L):
        d[i,:] = (x[(i + 1) % L,:] - x[i - 2,:]) * x[i - 1,:] - x[i,:] + F  
    return d.reshape(-1)

def Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau,rk45):
    
    sai = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),N)
    eta = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),N)
    
    x = np.zeros((N,L))
    y = np.zeros((N,dy))
    x0 = 10 + x0_amp*np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),1)
    x[0,] = x0

    for i in range(N-1):
        #print(i)
        if rk45:
            solver =  RK45(L96, t[i], x[i,],T,first_step=tau) 
            solver.step()
            x[i+1,] = solver.y + sai[i,]
            
        else:

            x[i+1,:] = x[i,:] + L96(t[i],x[i,:])*tau  + sai[i,:] 
        y[i+1,] = h(x[i+1,]) + eta[i+1,]
    return x,y#,time_RK45

def mse(x,x_true):
    x_mean = (x-x_true.reshape(AVG_SIM,N,L,1)).mean(axis=3)
    return ((x_mean*x_mean).sum(axis=2)).mean(axis=0)
#%%    
L = 9 # number of states
tau = 0.01 # timpe step 
T = 5 # final time in seconds
F = 10 # Force
N = int(T/tau) # number of time steps T = 20 s
dy = 3 #12 # number of states observed
observation_idx = [0,3,6]
# =============================================================================
# observation_idx = [0,2,4,6,8]
# =============================================================================
H = np.zeros((dy,L))

for i in range(dy):
    H[i,observation_idx[i]] = 1



rk45 = True

noise = np.sqrt(1e1) # noise level std
sigmma = noise/10 # Noise in the hidden state
sigmma0 = noise**1 # Noise in the initial state distribution
gamma = noise/10 # Noise in the observation
x0_amp = 1 # Amplifiying the initial state 
Noise = [noise,sigmma,sigmma0,gamma,x0_amp]


J = 250*1 # Number of ensembles EnKF
AVG_SIM = 10 # Number of Simulations to average over
reg = [0.01,0.01] # lambda_T lambda_f 
# reg = [0,0]

# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(32*1)
parameters['SAMPLE_SIZE'] = int(J) 
parameters['BATCH_SIZE'] = int(64/2)
parameters['LearningRate'] = 1e-4
parameters['ITERATION'] = int(1024) # *8 
parameters['Final_Number_ITERATION'] = int(64/1) #int(64) #ITERATION 
# =============================================================================
# parameters['Time_step'] = N
# =============================================================================

t = np.arange(0.0, tau*N, tau)


X_True = np.zeros((AVG_SIM,N,L))
Y_True = np.zeros((AVG_SIM,N,dy))
X0 = np.zeros((AVG_SIM,L,J))
for k in range(AVG_SIM):    
    x,y = Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau,rk45)
    X_True[k,] = x
    Y_True[k,] = y
    X0[k,] = 10 + x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))


# parameters['ITERATION'] = 1
X_EnKF  = EnKF(Y_True,X0,ML96,h,t,tau,Noise,rk45)
X_SIR  = SIR(Y_True,X0,ML96,h,t,tau,Noise,rk45)
X_OT = OT(Y_True,X0,parameters,ML96,h,t,tau,Noise,rk45)
X_OT_reg = OT_reg(Y_True,X0,parameters,ML96,h,t,tau,Noise,rk45,reg)

# X_EnKF = np.load('/Users/jarrah/Documents/SIAM_Data/L96/DATA_file_with_reg_last_ver.npz')['X_EnKF']
# X_SIR = np.load('/Users/jarrah/Documents/SIAM_Data/L96/DATA_file_with_reg_last_ver.npz')['X_SIR']
# X_OT = np.load('/Users/jarrah/Documents/SIAM_Data/L96/DATA_file_with_reg_last_ver.npz')['X_OT']
# X_OT_reg = np.load('/Users/jarrah/Documents/SIAM_Data/L96/DATA_file_with_reg_last_ver.npz')['X_OT_reg']
# parameters['Final_Number_ITERATION'] = int(64/16) #int(64) #ITERATION 
# # parameters['LearningRate'] = 1e-2
# parameters['ITERATION'] = int(1024*8) # *4 
# # parameters['NUM_NEURON'] =  int(32*2)
# # parameters['BATCH_SIZE'] = int(64/4)
# for i in [7]:
#     # x_OT  = OT(Y_True[i,].reshape(1,N,dy),X0[i,].reshape(1,L,-1),parameters,ML96,h,t,tau,Noise,rk45)
#     # X_OT[i,]  = x_OT.reshape(N,L,-1)
    
#     x_OT_reg  = OT_reg(Y_True[i,].reshape(1,N,dy),X0[i,].reshape(1,L,-1),parameters,ML96,h,t,tau,Noise,rk45,reg)
#     X_OT_reg[i,]  = x_OT_reg.reshape(N,L,-1)
    
#     # x_EnKF  = EnKF(Y_True[i,].reshape(1,N,dy),X0[i,].reshape(1,L,-1),ML96,h,t,tau,Noise,rk45)
    # X_EnKF[i,]  = x_EnKF.reshape(N,L,-1)


MSE_EnKF = mse(X_EnKF, X_True)
MSE_SIR = mse(X_SIR, X_True)
MSE_OT = mse(X_OT, X_True)
MSE_OT_reg = mse(X_OT_reg, X_True)
print(randint)
#%%
# i=0
# plot_particles = 100
# plt.figure(figsize=(12,6))      
# for l in range(9):
#     #for j in range(AVG_SIM):    
#     plt.subplot(3,3,l+1)   
# # =============================================================================
# #     for i in range(J):
# # =============================================================================
#     plt.plot(t,X_OT[i,:,l,:plot_particles],'C0',alpha = 0.1)
#     plt.plot(t,X_True[i,:,l],'k--')
#     plt.ylabel('X'+str(l+1))
#     if l==1:
#         plt.title('OT')
#     # if l>=6:
#     #     plt.xlabel('time')

# plot_particles = 100
# plt.figure(figsize=(12,6))      
# for l in range(9):
#     #for j in range(AVG_SIM):    
#     plt.subplot(3,3,l+1)   
# # =============================================================================
# #     for i in range(J):
# # =============================================================================
#     plt.plot(t,X_OT_reg[i,:,l,:plot_particles],'C0',alpha = 0.1)
#     plt.plot(t,X_True[i,:,l],'k--')
#     plt.ylabel('X'+str(l+1))
#     if l==1:
#         plt.title('OT_reg')
#     # if l>=6:
#     #     plt.xlabel('time')

# plt.figure(figsize=(6,6)) 
# plt.semilogy(t,MSE_EnKF,'g-.',label="EnKF",lw=2)
# plt.semilogy(t,MSE_SIR,'b-.',label="SIR",lw=2 )
# plt.semilogy(t,MSE_OT,'r--',label="OT",lw=2 )
# plt.semilogy(t,MSE_OT_reg,'m--',label=r"$OT_{reg}$",lw=2 )

# plt.xlabel('time')
# plt.ylabel('log(mse)')
# plt.legend()

# sys.exit()        
#%%
plot_particles = 100
j = 0
plt.figure(figsize=(12,6))   
for l in range(9):
    plt.subplot(3,3,l+1)
# =============================================================================
#     for i in range(J):
# =============================================================================
    plt.plot(t,X_EnKF[j,:,l,:plot_particles],'C0',alpha = 0.1)
    plt.plot(t,X_True[j,:,l],'k--')
    if l>=6:
        plt.xlabel('time')
    plt.ylabel('X'+str(l+1))
    if l==1:
        plt.title('EnKF')

plt.figure(figsize=(12,6)) 
for l in range(9):
    plt.subplot(3,3,l+1)
# =============================================================================
#     for i in range(J):
# =============================================================================
    plt.plot(t,X_SIR[j,:,l,:plot_particles],'C0',alpha = 0.1)
    plt.plot(t,X_True[j,:,l],'k--')
    plt.ylabel('X'+str(l+1))
    if l==1:
        plt.title('SIR')
    if l>=6:
        plt.xlabel('time')
    #plt.legend()

plt.figure(figsize=(12,6))      
for l in range(9):
    #for j in range(AVG_SIM):    
    plt.subplot(3,3,l+1)   
# =============================================================================
#     for i in range(J):
# =============================================================================
    plt.plot(t,X_OT[j,:,l,:plot_particles],'C0',alpha = 0.1)
    plt.plot(t,X_True[j,:,l],'k--')
    plt.ylabel('X'+str(l+1))
    if l==1:
        plt.title('OT')
    if l>=6:
        plt.xlabel('time')

plt.figure(figsize=(12,6))      
for l in range(9):
    #for j in range(AVG_SIM):    
    plt.subplot(3,3,l+1)   
# =============================================================================
#     for i in range(J):
# =============================================================================
    plt.plot(t,X_OT_reg[j,:,l,:plot_particles],'C0',alpha = 0.1)
    plt.plot(t,X_True[j,:,l],'k--')
    plt.ylabel('X'+str(l+1))
    if l==1:
        plt.title(r'$OT_{reg}$')
    if l>=6:
        plt.xlabel('time')
        
        
#%%
plt.figure(figsize=(6,6)) 
plt.semilogy(t,MSE_EnKF,'g-.',label="EnKF",lw=2)
plt.semilogy(t,MSE_SIR,'b-.',label="SIR",lw=2 )
plt.semilogy(t,MSE_OT,'r--',label="OT",lw=2 )
plt.semilogy(t,MSE_OT_reg,'m--',label=r"$OT_{reg}$",lw=2 )

plt.xlabel('time')
plt.ylabel('log(mse)')
plt.legend()
plt.show()

print(randint)

# sys.exit()
#%%
    
np.savez('DATA_file.npz',\
    time = t, Y_true = Y_True,X_true = X_True,Noise=Noise,\
    X_EnKF = X_EnKF , X_OT = X_OT , X_SIR = X_SIR,\
        MSE_EnKF = MSE_EnKF , MSE_OT=MSE_OT, MSE_SIR = MSE_SIR,\
        X_OT_reg = X_OT_reg , MSE_OT_reg = MSE_OT_reg,reg=reg)
  

    
 
