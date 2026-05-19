#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:27:36 2024

@author: jarrah
"""



import torch
import numpy as np
import time
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import sys
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


plt.close('all')
plt.rc('font', size=13)          # controls default text sizes
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

torch.manual_seed(101*0)
# =============================================================================
# class f_NN(nn.Module):
#         
#         def __init__(self, input_dim, hidden_dim):
#             super(f_NN, self).__init__()
#             self.input_dim = input_dim
#             self.hidden_dim = hidden_dim
#             self.activationSigmoid = nn.Sigmoid()
#             self.activationReLu = nn.ReLU()
#             self.layer_input = nn.Linear(self.input_dim[0]+self.input_dim[1], self.hidden_dim, bias=False)
#             self.layer_1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#             self.layer_out = nn.Linear(self.hidden_dim, 1, bias=True)
# 
#             
#         # Input is of size
#         def forward(self, x,y):
#             h = self.layer_input(torch.concat((x,y),dim=1))
#             h_temp = self.layer_1(self.activationReLu(h)) 
#             z = self.layer_out(self.activationReLu(h_temp) + h)  #+ 0.01*(x*x).sum(dim=1)
#             return z
# =============================================================================
# =============================================================================
# class f_NN(nn.Module):
#         
#         def __init__(self, input_dim, hidden_dim):
#             super(f_NN, self).__init__()
#             self.input_dim = input_dim
#             self.hidden_dim = hidden_dim
#             self.activationSigmoid = nn.Sigmoid()
#             self.activationReLu = nn.ReLU()
#             self.activationNonLinear = nn.Sigmoid()
#             self.activationELU = nn.ELU()
#             
#             self.layer_input = nn.Linear(self.input_dim[0]+self.input_dim[1], self.hidden_dim, bias=False)
#             self.linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
#             self.quad = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
#             self.cub = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
#             self.layer11 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#             self.layer12 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#             self.layer21 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#             self.layer22 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#             self.layerout = nn.Linear(self.hidden_dim, 1, bias=False)
#             
#             
#         # Input is of size
#         def forward(self, x,y):
#             
#             X = self.layer_input(torch.concat((x,y),dim=1))
#             
#             X = self.linear(X) #+ self.quad(X*X) + self.cub(X*X*X)
#             
#             xy = self.layer11(X)
#             xy = self.activationReLu(xy)
#             xy = self.layer12 (xy)
#             
#             X = self.activationReLu(xy+X)
# # =============================================================================
# #             X = self.activationReLu(xy+X+X*X)
# # =============================================================================
#             
#             xy = self.layer21(X)
#             xy = self.activationReLu(xy)
#             xy = self.layer22 (xy)
#             
#             X = self.layerout(self.activationReLu(xy+X))
#             xy = X
#             return xy
# =============================================================================
class f_NN(nn.Module):
        
        def __init__(self, input_dim, hidden_dim):
            super(f_NN, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.activationSigmoid = nn.Sigmoid()
            self.activationReLu = nn.ReLU()
            self.activationNonLinear = nn.Sigmoid()
            self.activationELU = nn.ELU()
            
            self.layer_input = nn.Linear(self.input_dim[0]+self.input_dim[1], self.hidden_dim, bias=False)
            self.linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.quad = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.cub = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.layer11 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer12 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer21 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer22 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer_out = nn.Linear(self.hidden_dim, 1, bias=False)
            
            
        # Input is of size
        def forward(self, x,y):
            
            X = self.layer_input(torch.concat((x,y),dim=1))
            
            xy = self.layer11(X)
            xy = self.activationELU(xy)
            xy = self.layer12 (xy)
            
            X = self.activationELU(xy+X)
            
            xy = self.layer21(X)
            xy = self.activationELU(xy)
            xy = self.layer22 (xy)
            
            xy = self.layer_out(self.activationELU(xy+X))
            return xy

class map_NN(nn.Module):
        
        def __init__(self, input_dim, hidden_dim):
            super(map_NN, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.activationSigmoid = nn.Sigmoid()
            self.activationReLu = nn.ReLU()
            self.activationNonLinear = nn.Sigmoid()
            
            self.layer_input = nn.Linear(self.input_dim[0]+self.input_dim[1], self.hidden_dim, bias=False)
            self.layer11 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer12 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer21 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer22 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer_out = nn.Linear(self.hidden_dim, input_dim[0], bias=False)
            
            
        # Input is of size
        def forward(self, x,y):
            
            X = self.layer_input(torch.concat((x,y),dim=1))
            
            xy = self.layer11(X)
            xy = self.activationReLu(xy)
            xy = self.layer12 (xy)
            
            X = self.activationReLu(xy+X)
            
            xy = self.layer21(X)
            xy = self.activationReLu(xy)
            xy = self.layer22 (xy)
            
            X = self.layer_out(self.activationReLu(xy+X))
            xy = X
            return xy

def init_weights(m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.xavier_normal_(m.weight)
            #torch.nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
            #torch.nn.init.kaiming_uniform_(m.weight,mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0.01)
                
def train(f,T,X_Train,Y_Train, iterations,learning_rate,batch_size,delta_T,delta_f):
        f.train()
        T.train()
        optimizer_T = torch.optim.Adam(T.parameters(), lr=learning_rate) 
        optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate)
        scheduler_f = ExponentialLR(optimizer_f, gamma=0.999) #set LR = 1e-1
        scheduler_T = ExponentialLR(optimizer_T, gamma=0.999) #set LR = 1e-1
       
        inner_iterations = 10
        Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0])].view(Y_Train.shape)
        for i in range(iterations):
            idx = torch.randperm(X_Train.shape[0])[:batch_size]
            X_train = X_Train[idx].clone().detach()
            Y_train = Y_Train[idx].clone().detach()
            Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)
            
            idx2 = torch.randperm(X_Train.shape[0])[:batch_size]
            X_train2 = X_Train[idx2].clone().detach()
      
            
            for j in range(inner_iterations):
                map_T = T.forward(X_train,Y_shuffled)
                f_of_map_T= f.forward(map_T,Y_shuffled) 
                #grad_f_of_map_T = torch.autograd.grad(f_of_map_T.sum(),map_T,create_graph=True)[0]
                
                map_T2 = T.forward(X_train2,Y_shuffled)
             
                reg =  nn.functional.elu(((map_T2 - map_T)*(-X_train2 + X_train)).sum(axis=1), alpha= 0.01 ).mean() #*1/batch_size 
# =============================================================================
#                 reg =  nn.functional.softplus(((map_T2 - map_T)*(-X_train2 + X_train)).sum(axis=1)).mean() #*1/batch_size 
# =============================================================================

                loss_T = -f_of_map_T.mean() +0.5*((X_train - map_T)*(X_train - map_T)).sum(axis=1).mean() + delta_T * reg
                optimizer_T.zero_grad()
                loss_T.backward()
                optimizer_T.step()
               
            f_of_y = f.forward(X_train,Y_train) 
            map_T = T.forward(X_train,Y_shuffled)
            f_of_map_T= f.forward(map_T,Y_shuffled) 
            
            # Compute the hessian
            reg2 = 0
            if delta_f != 0:
                laplacian = 0
                K_hessian = 4#batch_size
                # K_hessian = min(4,batch_size) # batch_size 32
                for kk in range(K_hessian): #range(batch_size):
                    x = X_train[kk].view(1,d)
                    y = Y_train[kk].view(1,dy)
                    hessian = torch.autograd.functional.hessian(f,(x,y),create_graph=True)
                    norm_hessian = torch.norm(hessian[0][0].reshape(d,d).diag())
                    laplacian += norm_hessian
                    
                    reg2 =  nn.functional.elu(laplacian, alpha= 0.01 )/K_hessian 
                    
            loss_f = -f_of_y.mean() + f_of_map_T.mean() + delta_f * reg2
            
            optimizer_f.zero_grad()
            loss_f.backward()
            optimizer_f.step()

            if  (i+1)==iterations or i%500==0:
                with torch.no_grad():
                    f_of_y = f.forward(X_Train,Y_Train) 
                    map_T = T.forward(X_Train,Y_Train_shuffled)
                    f_of_map_T= f.forward(map_T,Y_Train_shuffled) 
                    
                    loss = f_of_y.mean() - f_of_map_T.mean() + 0.5*((X_Train-map_T)*(X_Train-map_T)).sum(axis=1).mean()
                    #print(g.W.data)
                    print("Iteration: %d/%d, loss = %.4f" 
                          %(i+1,iterations,loss.item()))
                
            
             
            scheduler_f.step()
            scheduler_T.step()        

#%%
d = 1
dy =1
N = 1000
sigma = np.sqrt(1e-2)

x = torch.randn((N,d))

y = 0.5*x*x + sigma*torch.randn((N,dy))

# =============================================================================
# y = 3+sigma*torch.randn((N,d))
# =============================================================================
# =============================================================================
# y = 3+torch.rand((N,d))
# =============================================================================

# =============================================================================
# y = (torch.randint(0,2,(N,2))*2-1)*3 + sigma*torch.randn((N,d))
# =============================================================================

ITERS = int(1e3*5)
LR = 1e-3

INPUT_DIM = [d,dy]
NUM_NEURON = int(32)
BATCH_SIZE =32*2

def h_1D(x):
        return (0.5*x*x)
    
y_true = 1
xx = np.linspace(-3,3,100)
dx = 6./100
px = np.exp(-xx*xx/2) 
px = px/np.sum(px*dx)
pyx =  np.exp(-(y_true-h_1D(xx))*(y_true-h_1D(xx))/(2*sigma*sigma))
pxy = px*pyx
pxy = pxy/np.sum(pxy*dx)   


fontsize = 16

Lambda = [0,0.01,0.1]
plt.figure(figsize=(25,6))
for lamda in Lambda:
    Delta_T = lamda # regularization weight for T
    Delta_f = lamda # regularization weight for f


    start_time = time.time()
    
        
        
    f = f_NN(INPUT_DIM, NUM_NEURON)
    MAP_T = map_NN(INPUT_DIM, NUM_NEURON)
    
    MAP_T.apply(init_weights)
    f.apply(init_weights)
    
        
    train(f,MAP_T,x,y,ITERS,LR,BATCH_SIZE,Delta_T,Delta_f)
    
    x_transported = torch.randn((N*10,d))
    y_true = torch.ones_like(x_transported)  
    x_transported = MAP_T.forward(x_transported,y_true).detach().numpy()


    print("--- OT time : %s seconds ---" % (time.time() - start_time))



    x_plot = torch.linspace(-5,5,1000).view(-1,1)
    
    y_plot = torch.ones_like(x_plot)
    with torch.no_grad():
        f_plot = f(x_plot,y_plot)
    




    #%% Plot exact f
    
    y_true = 1
    xx = np.linspace(-3,3,1000)
    dx = 6./1000
    px = np.exp(-xx*xx/2) 
    px = px/np.sum(px*dx)
    pyx =  np.exp(-(y_true-h_1D(xx))*(y_true-h_1D(xx))/(2*sigma*sigma))
    pxy = px*pyx
    pxy = pxy/np.sum(pxy*dx)  
    
    sum_matrix = np.ones((px.size,px.size))
    sum_matrix = np.tril(sum_matrix)
    
    F_px = sum_matrix@(pxy*dx)
    F_pxy = sum_matrix@(px*dx)
    
    f_plot = f(torch.tensor(xx.reshape(-1,1),dtype=torch.float32),y_plot)
    
    
    F_inv_of_F_px = np.interp(F_px, F_pxy, xx, left=None, right=None, period=None)
    
    f_x_y_1 = sum_matrix@(F_inv_of_F_px*dx)


    
    plt.subplot(1,3,1)
    
    plt.xlabel('U',fontsize = fontsize)
    if lamda == 0:
# =============================================================================
#         plt.plot(xx,px,label=r"$P_U$",color='black',lw=2)
# =============================================================================
        plt.plot(xx,pxy,color='k',label=r"$P_{U|Y=1}$",lw=2)
        sns.kdeplot(data=x_transported[:,0],label=r'$OT_{(\lambda=0)}$',color="blue",linestyle='--',lw=2)
    elif lamda ==0.01:
        sns.kdeplot(data=x_transported[:,0],color='green',label=r'$OT_{(\lambda=0.01)}$',linestyle='-.',lw=2)
    else:
        sns.kdeplot(data=x_transported[:,0],color='red',label=r'$OT_{(\lambda=0.1)}$',linestyle='-.',lw=2)
    plt.legend(loc=0,fontsize = fontsize)
    
    plt.subplot(1,3,2)
    if lamda == 0:
        plt.plot(xx, f_x_y_1 - f_x_y_1.mean(),'k-',lw=2)#,label=r'exact'
        plt.plot(xx,0.5*xx*xx - f_plot.detach().numpy()[:,0] - (0.5*xx*xx - f_plot.detach().numpy()).mean(),'b--',lw=2)# ,label=r"$0.5\|U\|^2 - \phi(Y=1,U)$"
    elif lamda ==0.01:
        plt.plot(xx,0.5*xx*xx - f_plot.detach().numpy()[:,0] - (0.5*xx*xx - f_plot.detach().numpy()).mean(),'g-.',lw=2)# ,label=r"$0.5\|U\|^2 - \phi(Y=1,U)$",lw=2)
    else:
        plt.plot(xx,0.5*xx*xx - f_plot.detach().numpy()[:,0] - (0.5*xx*xx - f_plot.detach().numpy()).mean(),'r-.',lw=2)# ,label=r"$0.5\|U\|^2 - \phi(Y=1,U)$",lw=2)
        
    plt.xlabel('U',fontsize = fontsize)
    plt.ylabel(r'$0.5\|U\|^2 - \phi(Y=1,U)$',fontsize = fontsize)
# =============================================================================
#     plt.legend()
# =============================================================================

    T_plot = MAP_T(torch.tensor(xx.reshape(-1,1),dtype=torch.float32),y_plot)
    T_exact = np.interp(F_pxy, F_px, xx, left=None, right=None, period=None)
    plt.subplot(1,3,3)
    
    if lamda == 0:
        plt.plot(xx[:-1], T_exact[:-1],'k-',lw=2)#,label="exact map"
        plt.plot(xx, T_plot.detach().numpy()[:,0],'b--',lw=2)# ,label=r"$TY=1,U)$",lw=2)
    elif lamda ==0.01:
        plt.plot(xx, T_plot.detach().numpy()[:,0],'g-.',lw=2)# ,label=r"$T(Y=1,U)$",lw=2)
    else:
        plt.plot(xx, T_plot.detach().numpy()[:,0],'r-.',lw=2)# ,label=r"$T(Y=1,U)$"
    plt.xlabel('U',fontsize = fontsize)
    plt.ylabel(r"$T(Y=1,U)$",fontsize = fontsize)
# =============================================================================
#     plt.legend()
# =============================================================================
# =============================================================================
#     plt.suptitle(r'$\lambda_T = %0.2f, \lambda_f = %0.2f$'%(Delta_T,Delta_f))
# =============================================================================
    plt.show()


