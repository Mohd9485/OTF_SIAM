

import numpy as np
import matplotlib.pyplot as plt
import torch, math, time
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import sys

from EnKF import EnKF
from SIR import SIR
from OT_norm import OT
from OT_reg import OT_reg

# from scipy.integrate import odeint

# from sklearn.metrics import mean_squared_error
#%matplotlib auto

# =============================================================================
# np.random.seed(101)
# torch.manual_seed(101)
# =============================================================================

randint = np.random.randint(0,1000)
# randint = 592 # 732 #657 # 482 # 634 # 529 #768
randint = 101
print(randint)
np.random.seed(randint)
torch.manual_seed(randint)


plt.close('all')

# Choose h(x) here, the observation rule
def h(x):
    # return x[0,].reshape(dy,-1)
    return x[2,].reshape(dy,-1)
# =============================================================================
#     return x[::2,]
# =============================================================================

def L63(x, t):
    """Lorenz 96 model"""
    # Setting up vector
    #L = 3
    d = np.zeros_like(x)
    sigma = 10
    r = 28
    b = 8/3

    d[0] = sigma*(x[1]-x[0])
    d[1] = x[0]*(r-x[2])-x[1]
    d[2] = x[0]*x[1]-b*x[2]
    return d


def Gen_True_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau):
    eta = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),N)
    
    x = np.zeros((N,L))
    y = np.zeros((N,dy))
    # x0 = (2*torch.randint(0,2,(1,)).item()-1)*10 + np.random.multivariate_normal(np.zeros(L), np.eye(L),1)
    x0 = 5 + np.random.multivariate_normal(np.zeros(L), np.eye(L),1)
    x[0,] = x0

    
    for i in range(N-1):
        x[i+1,:] = x[i,:] + L63(x[i,:],t[i])*tau 
        y[i+1,] = h(x[i+1,]) + eta[i+1,]
    
    return x,y

def mse(x,x_true):
    x_mean = (x-x_true.reshape(AVG_SIM,N,L,1)).mean(axis=3)
    return ((x_mean*x_mean).sum(axis=2)).mean(axis=0)

#%%   
L = 3 # number of states
tau = 1e-2 # timpe step 
T = 2 # final time in seconds
N = int(T/tau) # number of time steps T = 20 s

dy = 1 # number of states observed
H = np.zeros((dy,L))
# =============================================================================
# H[0,0] = 1
# =============================================================================


noise = np.sqrt(1e1) # noise level std
sigmma = noise/10 # Noise in the hidden state
sigmma0 = noise**2 # Noise in the initial state distribution
gamma = noise/1 # Noise in the observation
x0_amp = 1 # Amplifiying the initial state 
Noise = [noise,sigmma,sigmma0,gamma,x0_amp]
Odeint = False


delta = [0.01,0.01] # lambda_T lambda_f 
J = int(1000/4) # Number of ensembles EnKF
AVG_SIM = 1 # Number of Simulations to average over

# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(32/1)
parameters['BATCH_SIZE'] = int(64/2)
parameters['LearningRate'] = 1e-3
parameters['ITERATION'] = int(1024/1) #1024*2 
parameters['Final_Number_ITERATION'] = int(64/1) #int(64*2) #ITERATION 




t = np.arange(0.0, tau*N, tau)
X_True = np.zeros((AVG_SIM,N,L))
Y_True = np.zeros((AVG_SIM,N,dy))
X0 = np.zeros((AVG_SIM,L,J))
for k in range(AVG_SIM):    
    x,y = Gen_True_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau)
    X_True[k,] = x
    Y_True[k,] = y
    X0[k,] = np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))


# data is AVG_SIM x N x L x J
X_EnKF = EnKF(Y_True,X0,L63,h,t,tau,Noise,Odeint)
X_SIR = SIR(Y_True,X0,L63,h,t,tau,Noise,Odeint)
X_OT = OT(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint) 
X_OT_reg = OT_reg(Y_True,X0,parameters,L63,h,t,tau,Noise,Odeint,delta)

# =============================================================================
# MSE_EnKF = mse(X_EnKF, X_True)
# MSE_SIR = mse(X_SIR, X_True)
# MSE_OT = mse(X_OT, X_True)
# MSE_OT_reg = mse(X_OT_reg, X_True)
# =============================================================================
print(randint)

#%%
# num_plot_state = 1 # number of state to plot
# # l = 4 # number of simulation to plot
# p = 100 # number of particles to plot

# for l in range(AVG_SIM):
#     plt.figure(figsize=(15,10))
#     for i in range(L):
#         plt.subplot(3,1,i+1)
#         plt.plot(t,X_EnKF[l,:,i,:p],'g',ls='none',marker='o',ms=4,alpha = 0.1)
#         plt.plot(t,X_True[l,:,i],'k--')
#         plt.ylabel('SIR')
    
#     plt.figure(figsize=(15,10))
#     for i in range(L):
#         plt.subplot(3,1,i+1)
#         plt.plot(t,X_SIR[l,:,i,:p],'b',ls='none',marker='o',ms=4,alpha = 0.1)
#         plt.plot(t,X_True[l,:,i],'k--')
#         plt.ylabel('SIR')
        
#     plt.figure(figsize=(15,10))
#     for i in range(L):
#         plt.subplot(3,1,i+1)
#         plt.plot(t,X_OT[l,:,i,:p],'r',ls='none',marker='o',ms=4,alpha = 0.1)
#         plt.plot(t,X_True[l,:,i],'k--')
#     plt.ylabel('OT')
    
#     plt.figure(figsize=(15,10))
#     for i in range(L):
#         plt.subplot(3,1,i+1)
#         plt.plot(t,X_OT_reg[l,:,i,:p],'m',ls='none',marker='o',ms=4,alpha = 0.1)
#         plt.plot(t,X_True[l,:,i],'k--')
#     plt.ylabel(r'$OT_{reg}$')

# sys.exit()
#%%
p = 100 # number of particles to plot
num_plot_state = 1 # number of state to plot
l=0

plt.figure(figsize=(15,10))
plt.subplot(5,1,1)
plt.plot(t,X_EnKF[l,:,num_plot_state,:p],'g',ls='none',marker='o',ms=4,alpha = 0.1)
plt.plot(t,X_True[l,:,num_plot_state],'k--',label='True state')
plt.ylabel('EnKF')
plt.legend()

plt.subplot(5,1,2)
plt.plot(t,X_SIR[l,:,num_plot_state,:p],'b',ls='none',marker='o',ms=4,alpha = 0.1)
plt.plot(t,X_True[l,:,num_plot_state],'k--')
plt.ylabel('SIR')


plt.subplot(5,1,3)
plt.plot(t,X_OT[l,:,num_plot_state,:p],'r',ls='none',marker='o',ms=4,alpha = 0.1)
plt.plot(t,X_True[l,:,num_plot_state],'k--')
plt.ylabel('OT')


plt.subplot(5,1,4)
plt.plot(t,X_OT_reg[l,:,num_plot_state,:p],'m',ls='none',marker='o',ms=4,alpha = 0.1)
plt.plot(t,X_True[l,:,num_plot_state],'k--')
plt.ylabel('OT_reg')
plt.xlabel('time')

# =============================================================================
# plt.subplot(5,1,5)
# plt.semilogy(t,MSE_EnKF,'g-.',label="EnKF")
# plt.semilogy(t,MSE_SIR,'b:',label="SIR" )
# plt.semilogy(t,MSE_OT,'r--',label="OT" )
# plt.semilogy(t,MSE_OT_reg,'m--',label=r"OT_{reg}" )
# plt.xlabel('time')
# plt.ylabel('log(mse)')
# plt.legend()
# =============================================================================

sys.exit()
#%%
np.savez('DATA_file.npz',\
    t = t, Noise=Noise,tau=tau,Odeint=Odeint, \
    X0 = X0, Y_true = Y_True,X_true = X_True, X_OT = X_OT , X_OT_reg = X_OT_reg)
