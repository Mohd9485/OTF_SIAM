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

def OT(Y,X0_const,parameters,A,h,t,tau,Noise,rk45):
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
    
    # OT networks parameters
    normalization = parameters['normalization']
    NUM_NEURON = parameters['NUM_NEURON']
    INPUT_DIM = parameters['INPUT_DIM']
    BATCH_SIZE =  parameters['BATCH_SIZE']
    LearningRate = parameters['LearningRate']
    ITERATION = parameters['ITERATION']
    final_iter = parameters['Final_Number_ITERATION']
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
#             self.hidden_dim = hidden_dim#*4
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
        
        # if ts >= 10 :
        #     optimizer_T = torch.optim.Adam(T.parameters(), lr=learning_rate*50) 
        #     optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate/10)
        #     scheduler_T = ExponentialLR(optimizer_T, gamma=0.8) #set LR = 1e-1
        #     scheduler_f = ExponentialLR(optimizer_f, gamma=0.9) #set LR = 1e-1
        # else:
        #     optimizer_T = torch.optim.Adam(T.parameters(), lr=learning_rate/1) 
        #     optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate/1)
        #     scheduler_T = ExponentialLR(optimizer_T, gamma=0.999) #set LR = 1e-1
        #     scheduler_f = ExponentialLR(optimizer_f, gamma=0.999) #set LR = 1e-1
        
        optimizer_T = torch.optim.Adam(T.parameters(), lr=learning_rate/1) 
        optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate/1)
        scheduler_T = ExponentialLR(optimizer_T, gamma=0.999) #set LR = 1e-1
        scheduler_f = ExponentialLR(optimizer_f, gamma=0.999) #set LR = 1e-1
        

        inner_iterations = 10
        Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0])].view(Y_Train.shape)
        for i in range(iterations):
            idx = torch.randperm(X1.shape[0])[:batch_size]
            X_train = X_Train[idx].clone().detach()
            Y_train = Y_Train[idx].clone().detach()
            
            Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)
            for j in range(inner_iterations):
                map_T = T.forward(X_train,Y_shuffled)
                f_of_map_T= f.forward(map_T,Y_shuffled) 
                # loss_T = - f_of_map_T.mean() + 0.5*((X_train-map_T)*(X_train-map_T)).sum(axis=1).mean()
                loss_T = - f_of_map_T.mean() + 0.5*((T.x_kf-map_T)*(T.x_kf-map_T)).sum(axis=1).mean()
                optimizer_T.zero_grad()
                loss_T.backward()
                optimizer_T.step()
                
               
                
            f_of_xy = f.forward(X_train,Y_train) 
            map_T = T.forward(X_train,Y_shuffled)
            f_of_map_T= f.forward(map_T,Y_shuffled) 
            loss_f = -f_of_xy.mean() + f_of_map_T.mean()
            optimizer_f.zero_grad()
            loss_f.backward()
            optimizer_f.step()
            
#             if ts >= 10 and i >= iterations/4 and i <= iterations*3/4 and (i - iterations/4) <= final_iter/2 :
# # =============================================================================
# #                 print(i,K/2,optimizer_T.param_groups[0]['lr'])
# # =============================================================================
#                 scheduler_f.step()
#                 scheduler_T.step()
#             elif ts < 10:
#                 scheduler_f.step()
#                 scheduler_T.step()
                
            if i<=1024:
                scheduler_f.step()
                scheduler_T.step()
           
            if  (i+1)==iterations:# or i%1024==0:
                with torch.no_grad():
                    f_of_xy = f.forward(X_Train,Y_Train) 
                    map_T = T.forward(X_Train,Y_Train_shuffled)
                    f_of_map_T = f.forward(map_T,Y_Train_shuffled) 
                    loss_f = f_of_xy.mean() - f_of_map_T.mean()
                    # loss = f_of_xy.mean() - f_of_map_T.mean() + 0.5*((X_Train-map_T)*(X_Train-map_T)).sum(axis=1).mean()
                    loss = f_of_xy.mean() - f_of_map_T.mean() + 0.5*((T.x_kf-map_T)*(T.x_kf-map_T)).sum(axis=1).mean()
                    print("Simu#%d/%d ,Time Step:%d/%d, Iteration: %d/%d, loss = %.8f" %(k+1,K,ts,Ts-1,i+1,iterations,loss.item()))
            

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
    SAVE_all_X_OT = np.zeros((AVG_SIM,N,J,L))

    for k in range(AVG_SIM):
        
        y = Y[k,]
        
        ITERS = ITERATION
        LR = LearningRate
        
        convex_f = NeuralNet(INPUT_DIM, NUM_NEURON)
        MAP_T = T_NeuralNet(INPUT_DIM, NUM_NEURON)
        
        convex_f.apply(init_weights_f)
        MAP_T.apply(init_weights)    
        torch.nn.init.orthogonal_(MAP_T.layer_input.weight)

        X0 = X0_const[k,].T
        X1 = np.zeros((J,L))
        Y1 = np.zeros((J,dy))
        x_OT = np.zeros((N,L))
        x_OT[0,:] = X0.mean(axis=0)
        SAVE_all_X_OT[k,0,:,:] = X0
        #plt.figure()
        for i in range(N-1):
           
            sai_train = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),J)
            if rk45:
                sai_train = sai_train.T
                solver =  RK45(A, t[i], (X0.T).reshape(-1),T,first_step=tau) 
                solver.step()
                X1 = (solver.y.reshape(L,J) + sai_train).T
            else: 
                X1 = X0 + ((A(t[i],X0.T)).reshape(L,J)*tau).T  + sai_train
                
            
            eta_train = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),J)
            Y1 = np.array(h(X1.T).T + eta_train)
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
            

            # dist = MultivariateNormal(torch.zeros(dy),gamma*gamma * torch.eye(dy))
            
            # y_hat = h(X1_train.T).T + dist.sample((X1_train.shape[0],))
            
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
            
            if ITERS > final_iter and i%1 == 0 :
                ITERS = int(ITERS/2)
            
            # if (i>0 and i% 100==0 and i<=200):
            #     ITERS = 1024
                
                
            Y1_true = y[i+1,:]
            Y1_true = torch.from_numpy(Y1_true)
            Y1_true = Y1_true.to(torch.float32)
            
            # Update X^(j) for the next time step
            X1_test = torch.from_numpy(X1).to(torch.float32).to(device)
            Y1_true = Y1_true.to(device)
            
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
            
    print("--- OT time : %s seconds ---" % (time.time() - start_time))
    return SAVE_all_X_OT.transpose((0,1,3,2)) 