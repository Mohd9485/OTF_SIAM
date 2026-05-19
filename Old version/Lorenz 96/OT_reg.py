#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:48:30 2024

@author: jarrah
"""

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import matplotlib.pyplot as plt
# =============================================================================
# from scipy.integrate import odeint
# =============================================================================
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.integrate import  RK45

def OT_reg(Y,X0_const,parameters,A,h,t,tau,Noise,rk45,delta):
    # Y is AVG_SIM x N x dy
    # X0 is AVG_SIM x L x J
    AVG_SIM = X0_const.shape[0]
    L = X0_const.shape[1]
    J = X0_const.shape[2]
    
    N = Y.shape[1]
    dy = Y.shape[2]
    
    noise = Noise[0]
    sigmma = Noise[1]# Noise in the hidden state
    sigmma0 = Noise[2] # Noise in the initial state distribution
    gamma = Noise[3] # Noise in the observation
    x0_amp = Noise[4]
    
    T = tau*N
    
    delta_T = delta[0]
    delta_f = delta[1]
    
    # OT networks parameters
    normalization = parameters['normalization']
    NUM_NEURON = parameters['NUM_NEURON']
    INPUT_DIM = parameters['INPUT_DIM']
    SAMPLE_SIZE = parameters['SAMPLE_SIZE']
    BATCH_SIZE =  parameters['BATCH_SIZE']
    LearningRate = parameters['LearningRate']
    ITERATION = parameters['ITERATION']
    Final_Number_ITERATION = parameters['Final_Number_ITERATION']
# =============================================================================
#     Time_step = parameters['Time_step']
# =============================================================================
    
    #device = torch.device('mps' if torch.has_mps else 'cpu') # M1 Chip
    device = torch.device('cpu')
    # NN , initialization and training    
# =============================================================================
#     class NeuralNet(nn.Module):
#         
#         def __init__(self, input_dim, hidden_dim):
#             super(NeuralNet, self).__init__()
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
        
        
    class NeuralNet(nn.Module):
            
            def __init__(self, input_dim, hidden_dim):
                super(NeuralNet, self).__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.activationSigmoid = nn.Sigmoid()
                # self.activation = nn.ReLU()
                self.activation = nn.ELU()
                
                self.layer_input = nn.Linear(self.input_dim[0]+self.input_dim[1], self.hidden_dim, bias=False)
                self.layer11 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layer12 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layer21 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layer22 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layer_out = nn.Linear(self.hidden_dim, 1, bias=False)
                
            # Input is of size
            def forward(self, x, y):
                X = self.layer_input(torch.concat((x,y),dim=1))
                
                xy = self.layer11(X)
                xy = self.activation(xy)
                xy = self.layer12 (xy)
                
                xy = self.activation(xy)+X
                
                xy = self.layer21(xy)
                xy = self.activation(xy)
                xy = self.layer22 (xy)
                
                xy = self.layer_out(self.activation(xy)+X)
                return xy
            
            
    class T_NeuralNet(nn.Module):
            
            def __init__(self, input_dim, hidden_dim):
                super(T_NeuralNet, self).__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.activationSigmoid = nn.Sigmoid()
                # self.activation = nn.ReLU()
                self.activation = nn.ELU()
                
                self.activationNonLinear = nn.Sigmoid()
                self.layer_input = nn.Linear(self.input_dim[0]+self.input_dim[1], self.hidden_dim, bias=False)
                self.layer11 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layer12 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layer21 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layer22 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layer_out = nn.Linear(self.hidden_dim, input_dim[0], bias=False)
                
                
                self.dist = MultivariateNormal(torch.zeros(self.input_dim[1]),gamma*gamma * torch.eye(self.input_dim[1]))
                self.x_kf = 0
            # Input is of size
            def forward(self, x, y):
                # EnKF settings
                # if i >= 10:
                y_hat = h(x.T).T + self.dist.sample((x.shape[0],))
                
                # m_hat = x.mean(axis=0,keepdims=True)
                # o_hat = y_hat.mean(axis=0,keepdims=True)
                
                # a = (x - m_hat)
                # b = (y_hat - o_hat)
                
                # C_xy = 1/x.shape[0] * a.T@b #np.matmul(a.transpose(),b)/J
                # C_yy = 1/x.shape[0] * b.T@b #np.matmul(b.transpose(),b)/J

                # K = C_xy @ torch.linalg.inv(C_yy + torch.eye(self.input_dim[1])*1e-6)#*gamma*gamma)
             
                self.x_kf = x + (K @ (y - y_hat).T).T 

                X = self.layer_input(torch.concat((self.x_kf,y),dim=1))
                
                xy = self.layer11(X)
                xy = self.activation(xy)
                xy = self.layer12 (xy)
                
                xy = self.activation(xy)+X
                
                xy = self.layer21(xy)
                xy = self.activation(xy)
                xy = self.layer22 (xy)
               
                xy = self.layer_out(self.activation(xy)+X) + self.x_kf
                # xy = self.layer_out(self.activationReLu(xy)) + self.layer_out_kf(x_kf)
                return xy
    
        
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.xavier_normal_(m.weight)
            # torch.nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
            # torch.nn.init.kaiming_uniform_(m.weight,mode='fan_out', nonlinearity='relu')
            
            # torch.nn.init.orthogonal_(m.weight)
            # torch.nn.init.uniform_(m.weight,-0.00001, 0.00001)
            # torch.nn.init.normal_(m.weight,0, 0.1)
            # m.weight.data.fill_(0.001*0)
            torch.nn.init.constant_(m.weight, 0)
            
            if m.bias is not None:
                m.bias.data.fill_(0.001)
                
    def init_weights_f(m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.xavier_normal_(m.weight)
            # torch.nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
            # torch.nn.init.kaiming_uniform_(m.weight,mode='fan_out', nonlinearity='relu')
            torch.nn.init.orthogonal_(m.weight)
            
            # torch.nn.init.uniform_(m.weight,-0.00001, 0.00001)
            # torch.nn.init.normal_(m.weight,0, 0.1)
            # m.weight.data.fill_(0.001*0)
            
            if m.bias is not None:
                m.bias.data.fill_(0.001)

    def train(f,T,X_Train,Y_Train, iterations,learning_rate,ts,Ts,batch_size,k,K):
        f.train()
        T.train()
        optimizer_T = torch.optim.Adam(T.parameters(), lr=learning_rate/1) 
        optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate/1)
# =============================================================================
#         optimizer_T = torch.optim.SGD(T.parameters(), lr=learning_rate,momentum=0.9) 
#         optimizer_f = torch.optim.SGD(f.parameters(), lr=learning_rate,momentum=0.9)
# =============================================================================
        scheduler_T = ExponentialLR(optimizer_T, gamma=0.999) #set LR = 1e-1
        scheduler_f = ExponentialLR(optimizer_f, gamma=0.999) #set LR = 1e-1
# =============================================================================
#         scheduler_f = StepLR(optimizer_f, step_size=50, gamma=0.9) #set LR = 1e-1
#         scheduler_T = StepLR(optimizer_T, step_size=50, gamma=0.9) #set LR = 1e-1
# =============================================================================
       
        inner_iterations = 10
        Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0])].view(Y_Train.shape)
        
        idx2 = torch.randperm(X_Train.shape[0])[:batch_size]
        
        
        for i in range(iterations):
            idx = torch.randperm(X1.shape[0])[:batch_size]
            #X_train = torch.tensor(X_Train[idx])
            #Y_train = torch.tensor(Y_Train[idx])
            X_train = X_Train[idx].clone().detach()
            Y_train = Y_Train[idx].clone().detach()
            X_train2 = X_Train[idx2].clone().detach()
            
            #X_train.requires_grad = True
            Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)
            #Y_shuffled.requires_grad = True
            for j in range(inner_iterations):
                map_T = T.forward(X_train,Y_shuffled)
                f_of_map_T= f.forward(map_T,Y_shuffled) 
                
                loss_T = - f_of_map_T.mean() + 0.5*((T.x_kf-map_T)*(T.x_kf-map_T)).sum(axis=1).mean() 
                
                
                map_T2 = T.forward(X_train2,Y_shuffled)
             
                reg =  nn.functional.elu(((map_T2 - map_T)*(-X_train2 + X_train)).sum(axis=1), alpha= 0.01 ).mean() #*1/batch_size 
                
                loss_T += delta_T * reg
                
                # loss_T = - f_of_map_T.mean() + 0.5*((X_train-map_T)*(X_train-map_T)).sum(axis=1).mean() + delta_T * reg
                
                + delta_T * reg
                optimizer_T.zero_grad()
                loss_T.backward()
                optimizer_T.step()
                
            f_of_xy = f.forward(X_train,Y_train) 
            map_T = T.forward(X_train,Y_shuffled)
            f_of_map_T= f.forward(map_T,Y_shuffled) 
            # Compute the hessian
            laplacian = 0
            K_hessian = min(4,batch_size) # batch_size 64
            for kk in range(K_hessian): #range(batch_size):
                
                x = X_train[kk].view(1,L)
                y = Y_train[kk].view(1,dy)
                hessian = torch.autograd.functional.hessian(f,(x,y),create_graph=True)
                norm_hessian = torch.norm(hessian[0][0].reshape(L,L).diag())
                laplacian += norm_hessian*norm_hessian
                
                
# =============================================================================
#                 X = X_train[kk].view(1,L).requires_grad_()
#                 f_lap = f(X,Y_train[kk].view(1,dy))
#                 grad_f = torch.autograd.grad(f_lap,X ,create_graph=True)
#                 print(torch.autograd.grad(grad_f[0][0][0],X[:,0],retain_graph=True,allow_unused=True))
#                 for l in range(L):
#                     print(torch.autograd.grad(grad_f[0][0][l],X[:,l],retain_graph=True, create_graph=True))
#                 ddf_x = torch.autograd.grad(torch.autograd.grad(f_lap,X,retain_graph=True, create_graph=True),X,retain_graph=True, create_graph=True)[0][0]
# 
#                 
#                 laplacian += ddf_x*ddf_x
# =============================================================================
      
            
                
# =============================================================================
#             reg2 =  nn.functional.softplus(laplacian)/batch_size
# =============================================================================
            reg2 =  nn.functional.elu(laplacian, alpha= 0.01 )/batch_size
            loss_f = -f_of_xy.mean() + f_of_map_T.mean() + delta_f * reg2

# =============================================================================
#             loss_f =f_of_xy.mean() - f_of_map_T.mean()
# =============================================================================
            optimizer_f.zero_grad()
            loss_f.backward()
            optimizer_f.step()
            
            if i<=1024:
                scheduler_f.step()
                scheduler_T.step()
                
            if  (i+1)==iterations:# or i%100==0:
                with torch.no_grad():
                    f_of_xy = f.forward(X_Train,Y_Train) 
                    map_T = T.forward(X_Train,Y_Train_shuffled)
                    f_of_map_T = f.forward(map_T,Y_Train_shuffled) 
                    loss_f = f_of_xy.mean() - f_of_map_T.mean()
                    # loss = f_of_xy.mean() - f_of_map_T.mean() + 0.5*((X_Train-map_T)*(X_Train-map_T)).sum(axis=1).mean()
                    loss = f_of_xy.mean() - f_of_map_T.mean() + 0.5*((T.x_kf-map_T)*(T.x_kf-map_T)).sum(axis=1).mean()
                    
                    #print(g.W.data)
                    print("Simu#%d/%d ,Time Step:%d/%d, Iteration: %d/%d, loss = %.4f" %(k+1,K,ts,Ts-1,i+1,iterations,loss.item()))
            

    def Normalization(X,Type = 'None'):
        ''' Normalize Date with type 'MinMax' out data between [0,1] or 'Mean' for mean 0 and std 1 '''
        if Type == 'None':
            return 0,0,X
        elif Type == 'Mean':
            Mean_X_training_data = torch.mean(X)
            Std_X_training_data = torch.std(X)
            return Mean_X_training_data , Std_X_training_data , (X - Mean_X_training_data)/Std_X_training_data
        elif Type == 'MinMax':
            Min = torch.min(X) 
            Max = torch.max(X)
            return Min , Max , (X-Min)/(Max-Min)

            
    def Transfer(M,S,X,Type='None'):
        '''Trasfer test Data to normalized data using knowledge of training data
        M = Mean/Min , S = Std/Max , X is data , Type = Mean/Min-Max Normalization '''
        if Type == 'None':
            return X
        elif Type == 'Mean':
            return (X - M)/S
        elif Type == 'MinMax':
            return (X - M)/(S - M)
        
    def deTransfer(M,S,X , Type = 'None'):
        ''' Detransfer the normalized data to the origin set
         M = Mean/Min , S = Std/Max , X is data , Type = Mean/Min-Max Normalization'''  
        if Type == 'None':
            return X
        elif Type == 'Mean':
            return X*S + M
        elif Type == 'MinMax':
            return X*(S - M) + M
    #
    start_time = time.time()
    SAVE_all_X_OT = np.zeros((AVG_SIM,N,SAMPLE_SIZE,L))
    # =============================================================================
    # SAVE_True_X_OT = np.zeros((AVG_SIM,N,L))
    # SAVE_True_Y_OT = np.zeros((AVG_SIM,N,dy))
    # =============================================================================
    
    for k in range(AVG_SIM):
        
        y = Y[k,]
        
    # =============================================================================
    #     SAVE_True_X_OT[k,] = x
    #     SAVE_True_Y_OT[k,] = y
    # =============================================================================
        
        ITERS = ITERATION
        LR = LearningRate
        
        convex_f = NeuralNet(INPUT_DIM, NUM_NEURON)
        MAP_T = T_NeuralNet(INPUT_DIM, NUM_NEURON)
        
        convex_f.apply(init_weights_f)
        MAP_T.apply(init_weights)    
        torch.nn.init.orthogonal_(MAP_T.layer_input.weight)
        
       
        #X0 = x0_amp*np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),SAMPLE_SIZE)
        X0 = X0_const[k,].T
        X1 = np.zeros((SAMPLE_SIZE,L))
        Y1 = np.zeros((SAMPLE_SIZE,dy))
        x_OT = np.zeros((N,L))
        x_OT[0,:] = X0.mean(axis=0)
        SAVE_all_X_OT[k,0,:,:] = X0
        #plt.figure()
        for i in range(N-1):
           
            sai_train = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),SAMPLE_SIZE)
            if rk45:
                sai_train = sai_train.T
# =============================================================================
#                 X1 = ((odeint(A, (X0.T).reshape(-1), t[i:i+2])[1,]).reshape(L,J) + sai_train).T
# =============================================================================
                solver =  RK45(A, t[i], (X0.T).reshape(-1),T,first_step=tau) 
                solver.step()
                X1 = (solver.y.reshape(L,J) + sai_train).T
# =============================================================================
#                 for j in range(J):
#                     X1[j,:] = odeint(A, X0[j,:], t[i:i+2])[1,] + sai_train[:,j]
# =============================================================================
            else: 
                X1 = X0 + ((A(t[i],X0.T)).reshape(L,J)*tau).T  + sai_train
            
            eta_train = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),SAMPLE_SIZE)
# =============================================================================
#             Y1 = h(X1.T).reshape(SAMPLE_SIZE,dy) + eta_train
# =============================================================================
            Y1 = np.array(h(X1.T).T + eta_train)
# =============================================================================
#             print( h(X1.T))
# =============================================================================
            X1_train = torch.from_numpy(X1)
            X1_train = X1_train.to(torch.float32)
            Y1_train = torch.from_numpy(Y1)
            Y1_train = Y1_train.to(torch.float32)
            X1_train = X1_train.to(device)
            Y1_train = Y1_train.to(device)
            
            #################################################
# =============================================================================
#             MX, SX, X1_train = Normalization(X1_train,Type = normalization)
#             MY, SY, Y1_train = Normalization(Y1_train,Type = normalization)
# =============================================================================
            
            m_hat = X1_train.mean(axis=0,keepdims=True)
            o_hat = Y1_train.mean(axis=0,keepdims=True)
            
            a = (X1_train - m_hat)
            b = (Y1_train - o_hat)
            
            C_xy = 1/X1_train.shape[0] * a.T@b #np.matmul(a.transpose(),b)/J
            C_yy = 1/X1_train.shape[0] * b.T@b #np.matmul(b.transpose(),b)/J

            K = C_xy @ torch.linalg.inv(C_yy + torch.eye(dy)*1e-6)#*gamma*gamma)
            
            train(convex_f,MAP_T,X1_train,Y1_train,ITERS,LR,i+1,N,BATCH_SIZE,k,AVG_SIM)
            
            # with torch.no_grad():
            #     print(MAP_T.layer11.bias.detach().sum(),MAP_T.layer21.bias.detach().sum())
            #     print(MAP_T.layer_out.weight.detach().max())
                
            if ITERS > Final_Number_ITERATION and i%1 == 0 :
                ITERS = int(ITERS/2)
                
# =============================================================================
#             if i>=100 and i%100 == 0:
#                 ITERS = ITERATION
# =============================================================================
                
            Y1_true = y[i+1,:]
            Y1_true = torch.from_numpy(Y1_true)
            Y1_true = Y1_true.to(torch.float32)
            
            # Update X^(j) for the next time step
            X1_test = torch.from_numpy(X1).to(torch.float32).to(device)
            Y1_true = Y1_true.to(device)
            
            #Y1_true = torch.ones_like(Y1_train)*Y1_true
            
            #################################################
# =============================================================================
#             X1_test = Transfer(MX, SX, X1_test,Type = normalization)
#             Y1_true = Transfer(MY, SY, Y1_true,Type = normalization)
# =============================================================================
            
            map_T = MAP_T.forward(X1_test, Y1_true*torch.ones((X1_test.shape[0],dy)))
            
            #################################################
# =============================================================================
#             map_T = deTransfer(MX, SX, map_T,normalization)
# =============================================================================
            
            if device.type == 'mps':
                X0 = map_T.cpu().detach().numpy()
            else:
                X0 = map_T.detach().numpy()
            
            x_OT[i+1,:] = (torch.mean(map_T,dim=0)).detach().numpy()
            SAVE_all_X_OT[k,i+1,:,:] = map_T.detach().numpy()
            
            
            #Y_shuffled = Y1_train[torch.randperm(Y1_train.shape[0])].view(Y1_train.shape)
            #map_T_plot = MAP_T.forward(X1_test, Y_shuffled)
            
    SAVE_all_X_OT = SAVE_all_X_OT.transpose((0,1,3,2))       
    print("--- OT time : %s seconds ---" % (time.time() - start_time))
    return SAVE_all_X_OT