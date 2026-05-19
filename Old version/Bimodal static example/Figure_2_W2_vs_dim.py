#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:14:56 2024

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
import ot
from torch.distributions.multivariate_normal import MultivariateNormal

plt.close('all')
plt.rc('font', size=13)          # controls default text sizes
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fontsize = 16
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
            # self.linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.quad = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.cub = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.layer11 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer12 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer21 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer22 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            # self.layer31 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            # self.layer32 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
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
            
            return self.layer_out(self.activationELU(xy+X))

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
            self.layer31 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer32 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
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
            
            return self.layer_out(self.activationReLu(xy+X))

def init_weights(m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.xavier_normal_(m.weight)
            # torch.nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
            # torch.nn.init.kaiming_uniform_(m.weight,mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                
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
                reg = 0
                if delta_T != 0:
                    map_T2 = T.forward(X_train2,Y_shuffled)
                 
                    reg =  nn.functional.elu(((map_T2 - map_T)*(-X_train2 + X_train)).sum(axis=1), alpha= 0.01 ).mean() #*1/batch_size 
# =============================================================================
#                 reg =  nn.functional.softplus(((map_T2 - map_T)*(-X_train2 + X_train)).sum(axis=1)).mean() #*1/batch_size 
# =============================================================================

                loss_T = -f_of_map_T.mean() + 0.5*((X_train - map_T)*(X_train - map_T)).sum(axis=1).mean() + delta_T * reg
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
                # K_hessian = min(4,batch_size) # batch_size 64
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

            if  (i+1)==iterations :#or i==0:#i%500==0:
                with torch.no_grad():
                    f_of_y = f.forward(X_Train,Y_Train) 
                    map_T = T.forward(X_Train,Y_Train_shuffled)
                    f_of_map_T= f.forward(map_T,Y_Train_shuffled) 
                    
                    loss = f_of_y.mean() - f_of_map_T.mean() + 0.5*((X_Train-map_T)*(X_Train-map_T)).sum(axis=1).mean()
                   
                    print("Iteration: %d/%d, loss = %.4f" 
                          %(i+1,iterations,loss.item()))
                
            
            # if i >= 500*0 and i<= 5000: 
            scheduler_f.step()
            scheduler_T.step()        


def h(x):
        return (0.5*x*x)
#%%
ITERS = int(1e4/2)
# ITERS = int(1e1)
LR = 1e-3

NUM_NEURON = int(32*3)
BATCH_SIZE = int(32*1)
N = 5000

AVG_SIM = 10

sigma = np.sqrt(1e-2)

D = [1,2,5,8,10,15,20]

Lambda = [0,0.01,0.1]


 
y_true = 1
xx = np.linspace(-3,3,100)
dx = 6./100
px = np.exp(-xx*xx/2) 
px = px/np.sum(px*dx)
pyx =  np.exp(-(y_true-h(xx))*(y_true-h(xx))/(2*sigma*sigma))
pxy = px*pyx
pxy = pxy/np.sum(pxy*dx)   



rng = np.random.default_rng()


N_true = int(1e5)
X_true = np.zeros((N_true,max(D)))
for j in range(max(D)):
    print(j)
    x_SIR = np.random.multivariate_normal(np.zeros(1),np.eye(1),N_true).T
    
    
    W = np.sum((y_true - h(x_SIR).T)*(y_true - h(x_SIR).T),axis=1)/(2*sigma*sigma)
    W = W - np.min(W)
    W = np.exp(-W).T
    W = W/np.sum(W)
    
    #x_SIR[k,i+1,0,] = rng.choice(x_SIR[k,i+1,0,], J, p = W[k,i+1,0,])
    index = rng.choice(np.arange(N_true), N_true, p = W)
    X_true[:,j] = x_SIR[:,index].reshape(N_true)  



distance_ot = {}
x_otf = {}
# y_save = {}
time_save = {}
for lamda in Lambda:
    distance_ot[str(lamda)] = []
    x_otf[str(lamda)] = {}
    time_save[str(lamda)] = []   


     
for dim in D:
    
    d = dim
    dy = dim


    INPUT_DIM = [d,dy]
    
    dist_normal = MultivariateNormal(torch.zeros(d), torch.eye(d))
    dist = MultivariateNormal(torch.zeros(dy),sigma*sigma * torch.eye(dy))
    
    x = dist_normal.sample((N,))
    
    y = h(x) + dist.sample((N,)) #sigma*torch.randn((N,dy))
    
    # y_save[str(dim)] = y
    
    for lamda in Lambda:
        print('dim = ',dim, ', lambda = ',lamda)
        Delta_T = lamda # regularization weight for T
        Delta_f = lamda # regularization weight for f
    
        w2 = 0
        track_time = 0
        for k in range(AVG_SIM): 
                
            f = f_NN(INPUT_DIM, NUM_NEURON)
            MAP_T = map_NN(INPUT_DIM, NUM_NEURON)
            
            MAP_T.apply(init_weights)
            f.apply(init_weights)
                
            start_time = time.time()
            
            train(f,MAP_T,x,y,ITERS,LR,BATCH_SIZE,Delta_T,Delta_f)

            x_transported = MAP_T.forward(x,torch.ones_like(x)).detach().numpy()
            
            track_time += (time.time() - start_time)
            
            print("--- OT time : %s seconds ---" % (time.time() - start_time))
            
            # compute W2
            p_true=int(1e3) # number of particles used to compute W-2
            # Compute the cost matrix (usually the Euclidean distance matrix)
            M_ot =  ot.dist(X_true[:p_true,:dim], x_transported) 
    
            # Uniform weights if distributions are unweighted
            a = np.ones(p_true) / p_true # Uniform weights for X
            b = np.ones(N) / N # Uniform weights for Y
    
            # Compute the Wasserstein distance (emd2 returns the squared distance)
            w2 += np.sqrt(ot.emd2(a, b, M_ot))
        distance_ot[str(lamda)].append(w2/AVG_SIM)
        x_otf[str(lamda)][str(dim)+'_'+str(k)] = x_transported
        time_save[str(lamda)].append(track_time/AVG_SIM)
        
#%%
X_SIR = {}
X_EnKF = {}
distance_sir = []
distance_enkf = []
y_save = {}
time_save = {}
time_save['sir'] = []
time_save['enkf'] = []    

for dim in D:
    print("dim = : ", dim)
    d = dim
    dy = dim
    
    dist_normal = MultivariateNormal(torch.zeros(d), torch.eye(d))
    dist = MultivariateNormal(torch.zeros(dy),sigma*sigma * torch.eye(dy))
    
    x = dist_normal.sample((N,))
    
    y = h(x) + dist.sample((N,)) #sigma*torch.randn((N,dy))
    
    y_save[str(dim)] = y
    
    w2_sir = 0
    w2_enkf = 0
    track_time_sir = 0
    track_time_enkf = 0
    for k in range(AVG_SIM): 
        # SIR
        start_time = time.time()
        
        x_SIR = np.random.multivariate_normal(np.zeros(dim),np.eye(dim),N).T
        
        
        W = np.sum((y_true - h(x_SIR).T)*(y_true - h(x_SIR).T),axis=1)/(2*sigma*sigma)
        W = W - np.min(W)
        W = np.exp(-W).T
        W = W/np.sum(W)
        
        #x_SIR[k,i+1,0,] = rng.choice(x_SIR[k,i+1,0,], J, p = W[k,i+1,0,])
        index = rng.choice(np.arange(N), N, p = W)
        X_SIR[str(dim)+'_'+str(k)] = x_SIR[:,index].T
        
        track_time_sir += (time.time() - start_time)
        
        print("--- SIR time : %s seconds ---" % (time.time() - start_time))
        
        
        # EnKF
        
        x_hatEnKF = x.detach().numpy()  
        y_hatEnKF = y.detach().numpy()
        
        start_time = time.time()
        X_hat = x_hatEnKF.mean(axis=0,keepdims=True)
        #o_hatEnKF = (x_hatEnKF**3).mean(axis=1)
        Y_hat = y_hatEnKF.mean(axis=0,keepdims=True)
        
        a = (x_hatEnKF - X_hat)
        b = (y_hatEnKF - Y_hat)
        
        C_xy = 1/N * a.T@b 
        C_yy = 1/N * b.T@b 
        K_EnKF = np.matmul(C_xy,np.linalg.inv(C_yy + np.eye(dy)*1e-4))
        X_EnKF[str(dim)+'_'+str(k)] = x_hatEnKF + ((y_true - y_hatEnKF)@K_EnKF.T) 
    
        track_time_enkf += (time.time() - start_time)
        
        print("--- EnKF time : %s seconds ---" % (time.time() - start_time))

        
        # compute W2 SIR
        p_true=int(1e3) # number of particles used to compute W-2
        # Compute the cost matrix (usually the Euclidean distance matrix)
        M_sir =  ot.dist(X_true[:p_true,:dim], X_SIR[str(dim)+'_'+str(k)]) 
    
        # Uniform weights if distributions are unweighted
        a = np.ones(p_true) / p_true # Uniform weights for X
        b = np.ones(N) / N # Uniform weights for Y
    
        # Compute the Wasserstein distance (emd2 returns the squared distance)
        
        w2_sir += np.sqrt(ot.emd2(a, b, M_sir))
        
        # compute W2 EnKF
        p_true=int(1e3) # number of particles used to compute W-2
        # Compute the cost matrix (usually the Euclidean distance matrix)
        M_enkf =  ot.dist(X_true[:p_true,:dim], X_EnKF[str(dim)+'_'+str(k)]) 
    
        # Uniform weights if distributions are unweighted
        a = np.ones(p_true) / p_true # Uniform weights for X
        b = np.ones(N) / N # Uniform weights for Y
    
        # Compute the Wasserstein distance (emd2 returns the squared distance)
        w2_enkf += np.sqrt(ot.emd2(a, b, M_enkf))
    
    distance_sir.append(w2_sir/AVG_SIM)   
    distance_enkf.append(w2_enkf/AVG_SIM)         
    time_save['sir'].append(track_time_sir/AVG_SIM)
    time_save['enkf'].append(track_time_enkf/AVG_SIM)
#%%
plt.figure(figsize=(12,6))    
for lamda in x_otf.keys():
    print(lamda)
    if lamda == '0.0' or lamda == '0':
        plt.semilogy(D,distance_ot[str(lamda)],'v--',color="blue",label=r"$OT_{(\lambda=0)}$",lw=2)
    elif lamda == '0.1':
        plt.semilogy(D,distance_ot[str(lamda)],'o-.',color="red",label=r'$OT_{(\lambda=0.1)}$',lw=2)
    elif lamda == '0.01':
        plt.semilogy(D,distance_ot[str(lamda)],'s-.',color="green",label=r'$OT_{(\lambda=0.01)}$',lw=2)

plt.semilogy(D,distance_enkf,'D:',color="C4",label=r"EnKF",lw=2.5)
plt.semilogy(D,distance_sir,'^:',color="C5",label=r"SIR",lw=2.5)

plt.xlabel(r'$dim$',fontsize=fontsize)
plt.ylabel(r'$W_2$',fontsize=fontsize)
plt.legend(fontsize=fontsize)