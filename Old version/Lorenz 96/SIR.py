import numpy as np
import time
# =============================================================================
# from scipy.integrate import odeint
# =============================================================================
from scipy.integrate import  RK45
def SIR(Y,X0,A,h,t,tau,Noise,rk45):
    # Y is AVG_SIM x N x dy
    # X0 is AVG_SIM x L x J
    AVG_SIM = X0.shape[0]
    L = X0.shape[1]
    J = X0.shape[2]
    
    N = Y.shape[1]
    dy = Y.shape[2]
    
    noise = Noise[0]
    sigmma = Noise[1]# Noise in the hidden state
    sigmma0 = Noise[2] # Noise in the initial state distribution
    gamma = Noise[3] # Noise in the observation
    x0_amp = Noise[4]
    
    T = tau*N
    
    start_time = time.time()
    x_SIR =  np.zeros((AVG_SIM,N,L,J))

    rng = np.random.default_rng()
    for k in range(AVG_SIM):
        x_SIR[k,0,] = X0[k,]
        y = Y[k,]
        
        for i in range(N-1):
            sai_SIR = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),J).transpose()
            if rk45:
                solver =  RK45(A, t[i], x_SIR[k,i,].reshape(-1),T,first_step=tau) 
                solver.step()
                x_SIR[k,i+1,] = solver.y.reshape(L,J) + sai_SIR
            else: 
                x_SIR[k,i+1,] = x_SIR[k,i,]+ A(t[i],x_SIR[k,i,]).reshape(L,J)*tau + sai_SIR
                
            W = np.sum((y[i+1,] - h(x_SIR[k,i+1,]).T)*(y[i+1] - h(x_SIR[k,i+1,]).T),axis=1)/(2*gamma*gamma)
            
            W = W - np.min(W)
            W = np.exp(-W).T
            W = W/np.sum(W)
            
            index = rng.choice(np.arange(J), J, p = W)
            x_SIR[k,i+1,] = x_SIR[k,i+1,:,index].T

    print("--- SIR time : %s seconds ---" % (time.time() - start_time))
    return x_SIR