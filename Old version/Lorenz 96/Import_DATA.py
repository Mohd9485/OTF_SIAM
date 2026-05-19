import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib

plt.close('all')

plt.rc('font', size=19)          # controls default text sizes
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
def relu(x):
    a = 0
    return np.maximum(x,a)
def mse(x,x_true):
    x_mean = (x-x_true.reshape(AVG_SIM,N,L,1)).mean(axis=3)
    return ((x_mean*x_mean).sum(axis=2)).mean(axis=0)

load = np.load('DATA_file.npz')

data = {}
for key in load:
    print(key)
    data[key] = load[key]
    
    
t = data['time']
X_true = data['X_true']
Y_true = data['Y_true']
X_EnKF = data['X_EnKF']
X_SIR = data['X_SIR']
X_OT = data['X_OT']
X_OT_reg = data['X_OT_reg']
# =============================================================================
# X_EnKF_PF = data['X_EnKF_PF']
# X_WEKF = data['X_WEKF']
# =============================================================================

# MSE_EnKF = data['MSE_EnKF']
# MSE_OT = data['MSE_OT']
# MSE_OT_reg = data['MSE_OT_reg']
# MSE_SIR = data['MSE_SIR']
# =============================================================================
# MSE_EnKF_PF = data['MSE_EnKF_PF']
# MSE_WEKF = data['MSE_WEKF']
# =============================================================================


reg = data['reg']




#%%
AVG_SIM = X_OT.shape[0]
J = X_EnKF.shape[3]
SAMPLE_SIZE = X_OT.shape[3]
L = X_true.shape[2]
N = len(t)
num_plot = int(np.sqrt(L))

MSE_EnKF = mse(X_EnKF, X_true)
MSE_SIR = mse(X_SIR, X_true)
MSE_OT = mse(X_OT, X_true)
MSE_OT_reg = mse(X_OT_reg, X_true)

#%%
j=0

num_plot_state = 0
x_lim = 25
plt.figure(figsize=(8,9))
plt.subplot(4,1,1)
plt.plot(t,X_EnKF[j,:,num_plot_state,:],'C4',alpha = 0.1)
plt.plot(t,X_true[j,:,num_plot_state],'k--',label='True state',lw=2)
plt.ylabel('EnKF')
plt.ylim(-x_lim,x_lim)
plt.legend(loc=4)
ax = plt.gca()
ax.get_xaxis().set_visible(False)

plt.subplot(4,1,2)
plt.plot(t,X_SIR[j,:,num_plot_state,:],'C5',alpha = 0.1)
plt.plot(t,X_true[j,:,num_plot_state],'k--',lw=2)
plt.ylabel('SIR')
plt.ylim(-x_lim,x_lim)
ax = plt.gca()
ax.get_xaxis().set_visible(False)

plt.subplot(4,1,3)
plt.plot(t,X_OT[j,:,num_plot_state,:],'b',alpha = 0.1)
plt.plot(t,X_true[j,:,num_plot_state],'k--',lw=2)
plt.ylabel(r'$OT~(\lambda=0)$')
plt.xlabel('time')
plt.ylim(-x_lim,x_lim)
ax = plt.gca()
ax.get_xaxis().set_visible(False)

plt.subplot(4,1,4)
plt.plot(t,X_OT_reg[j,:,num_plot_state,:],'g',alpha = 0.1)
plt.plot(t,X_true[j,:,num_plot_state],'k--',lw=2)
plt.ylabel(r'$OT~(\lambda=0.01)$')
plt.xlabel('time')
plt.xlabel('time')
plt.ylim(-x_lim,x_lim)


#%%
num_plot_state = 1
# x_lim = 25

plt.figure(figsize=(8,9))
plt.subplot(4,1,1)
plt.plot(t,X_EnKF[j,:,num_plot_state,:],'C4',alpha = 0.1)
plt.plot(t,X_true[j,:,num_plot_state],'k--',lw=2)#,label='True state')
# =============================================================================
# plt.ylabel('EnKF')
# =============================================================================
plt.ylim(-x_lim,x_lim)
# =============================================================================
# plt.legend(loc=1)
# =============================================================================
ax = plt.gca()
ax.get_xaxis().set_visible(False)

plt.subplot(4,1,2)
plt.plot(t,X_SIR[j,:,num_plot_state,:],'C5',alpha = 0.1)
plt.plot(t,X_true[j,:,num_plot_state],'k--',lw=2)
# =============================================================================
# plt.ylabel('SIR')
# =============================================================================
plt.ylim(-x_lim,x_lim)
ax = plt.gca()
ax.get_xaxis().set_visible(False)

plt.subplot(4,1,3)
plt.plot(t,X_OT[j,:,num_plot_state,:],'b',alpha = 0.1)
plt.plot(t,X_true[j,:,num_plot_state],'k--',lw=2)
# =============================================================================
# plt.ylabel('OT')
# =============================================================================
plt.ylim(-x_lim,x_lim)
ax = plt.gca()
ax.get_xaxis().set_visible(False)

plt.subplot(4,1,4)
plt.plot(t,X_OT_reg[j,:,num_plot_state,:],'g',alpha = 0.1)
plt.plot(t,X_true[j,:,num_plot_state],'k--',lw=2)
# =============================================================================
# plt.ylabel(r'$OT_{reg}$')
# =============================================================================
plt.xlabel('time')
plt.ylim(-x_lim,x_lim)

#%%
num_plot_state = 2
# x_lim = 65

plt.figure(figsize=(8,9))
plt.subplot(4,1,1)
plt.plot(t,X_EnKF[j,:,num_plot_state,:],'C4',alpha = 0.1)
plt.plot(t,X_true[j,:,num_plot_state],'k--',lw=2)#,label='True state')
# =============================================================================
# plt.ylabel('EnKF')
# =============================================================================
plt.ylim(-x_lim,x_lim)
# =============================================================================
# plt.legend(loc=1)
# =============================================================================
ax = plt.gca()
ax.get_xaxis().set_visible(False)

plt.subplot(4,1,2)
plt.plot(t,X_SIR[j,:,num_plot_state,:],'C5',alpha = 0.1)
plt.plot(t,X_true[j,:,num_plot_state],'k--',lw=2)
# =============================================================================
# plt.ylabel('SIR')
# =============================================================================
plt.ylim(-x_lim,x_lim)
ax = plt.gca()
ax.get_xaxis().set_visible(False)

plt.subplot(4,1,3)
plt.plot(t,X_OT[j,:,num_plot_state,:],'b',alpha = 0.1)
plt.plot(t,X_true[j,:,num_plot_state],'k--',lw=2)
# =============================================================================
# plt.ylabel('OT')
# =============================================================================
plt.ylim(-x_lim,x_lim)
ax = plt.gca()
ax.get_xaxis().set_visible(False)

plt.subplot(4,1,4)
plt.plot(t,X_OT_reg[j,:,num_plot_state,:],'g',alpha = 0.1)
plt.plot(t,X_true[j,:,num_plot_state],'k--',lw=2)
# =============================================================================
# plt.ylabel(r'$OT_{reg}$')
# =============================================================================
plt.xlabel('time')
plt.ylim(-x_lim,x_lim)

#%%
plt.figure(figsize=(8,9))

plt.semilogy(t,MSE_OT,'b--',label=r"$OT_{(\lambda=0)}$" ,lw=2)
plt.semilogy(t,MSE_OT_reg,'g-.',label=r"$OT_{(\lambda=0.01)}$" ,lw=2)
plt.semilogy(t,MSE_EnKF,':',color='C4',label=r"$EnKF$",lw=2.5)
plt.semilogy(t,MSE_SIR,':',color='C5',label=r"$SIR$" ,lw=2.5)


# =============================================================================
# plt.semilogy(t,MSE_WEKF,'m:',label="WEKF" )
# plt.semilogy(t,MSE_EnKF_PF,'c:',label="EnKF_PF" )
# =============================================================================
plt.xlabel('time')
plt.ylabel('MSE')
plt.legend(loc=0)
