import ot
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib
from EnKF import EnKF
from SIR import SIR
import seaborn as sns 

plt.close('all')

plt.rc('font', size=19)          # controls default text sizes
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
def relu(x):
    a = 0
    return np.maximum(x,a)

# Choose h(x) here, the observation rule
def h(x):
    # return x[0,].reshape(dy,-1)
    return x[2,].reshape(dy,-1)
# =============================================================================
#     return x[::2,]
# =============================================================================

def L63(x, t):
    """Lorenz 63 model"""
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

def mse(x,x_true):
    x_mean = (x-x_true.reshape(AVG_SIM,N,L,1)).mean(axis=3)
    return ((x_mean*x_mean).sum(axis=2)).mean(axis=0)
#%%

# data = np.load('DATA_file_keep_100sims.npz')
data = np.load('DATA_file.npz')

for key in data:
    print(key)
    # globals()[key] = data[key]

t = data['t']
Noise = data['Noise']
tau = data['tau']
Odeint = data['Odeint']
X0 = data['X0']
Y_true = data['Y_true']
X_true = data['X_true']
X_OT = data['X_OT']
X_OT_reg = data['X_OT_reg']

#%%
# save = 5
# np.savez('DATA_file_subset.npz',\
#     t = t, Noise=Noise,tau=tau,Odeint=Odeint, \
#     X0 = X0[:save,], Y_true = Y_true[:save,],X_true = X_true[:save,], X_OT = X_OT[:save,] , X_OT_reg = X_OT_reg[:save,])


#%%
AVG_SIM = X_OT.shape[0]
N = X_OT.shape[1]
L = X_OT.shape[2]
J = X_OT.shape[3]
dy = Y_true.shape[2]
#%%
X_SIR = SIR(Y_true,X0,L63,h,t,tau,Noise,Odeint)
X_EnKF = EnKF(Y_true,X0,L63,h,t,tau,Noise,Odeint)

MSE_EnKF = mse(X_EnKF, X_true)
MSE_SIR = mse(X_SIR, X_true)
MSE_OT = mse(X_OT, X_true)
MSE_OT_reg = mse(X_OT_reg, X_true)

# sys.exit()

#%% compute W2 for big AVG_SIM
distance_EnKF = []
distance_SIR = []
distance_OT = []
distance_OT_reg = []

p_true=int(1e3) # number of particles used to compute W-2

true_particle = int(1e5)

X0_true = np.zeros((AVG_SIM,L,true_particle))
for k in range(AVG_SIM):    
    X0_true[k,] =  np.transpose(np.random.multivariate_normal(np.zeros(L),Noise[2]*Noise[2] * np.eye(L),true_particle))


distance_enkf = {}
distance_sir = {}
distance_ot = {}
distance_ot_reg = {}


for i in range(N):
    distance_enkf[str(i)] = []
    distance_sir[str(i)] = []
    distance_ot[str(i)] = []
    distance_ot_reg[str(i)] = []
    
for k in range(AVG_SIM): 
    
    X_true_dist = SIR(Y_true[k,].reshape(1,N,dy),X0_true[k,].reshape(1,L,true_particle),L63,h,t,tau,Noise,Odeint)
    print("Sim: ",k)
    
    for i in range(N):
        # Compute the cost matrix (usually the Euclidean distance matrix)
        M_enkf =  ot.dist(X_true_dist[0,i,:,:p_true].T, X_EnKF[k,i,].T) 
        M_sir =  ot.dist(X_true_dist[0,i,:,:p_true].T, X_SIR[k,i,].T) 
        M_ot =  ot.dist(X_true_dist[0,i,:,:p_true].T, X_OT[k,i,].T) 
        M_ot_reg =  ot.dist(X_true_dist[0,i,:,:p_true].T, X_OT_reg[k,i,].T) 

        # Uniform weights if distributions are unweighted
        a = np.ones(p_true) / p_true # Uniform weights for X
        b = np.ones(J) / J # Uniform weights for Y

        # Compute the Wasserstein distance (emd2 returns the squared distance)
        distance_enkf[str(i)].append(np.sqrt(ot.emd2(a, b, M_enkf)))
        distance_sir[str(i)].append(np.sqrt(ot.emd2(a, b, M_sir)))
        distance_ot[str(i)].append(np.sqrt(ot.emd2(a, b, M_ot)))
        distance_ot_reg[str(i)].append(np.sqrt(ot.emd2(a, b, M_ot_reg)))
        
distance_EnKF = np.zeros((N,AVG_SIM))
distance_SIR = np.zeros_like(distance_EnKF)
distance_OT = np.zeros_like(distance_EnKF)
distance_OT_reg = np.zeros_like(distance_EnKF)

for i in range(N):
    distance_EnKF[i,] = np.array(distance_enkf[str(i)])
    distance_SIR[i,] = np.array(distance_sir[str(i)])
    distance_OT[i,] = np.array(distance_ot[str(i)])
    distance_OT_reg[i,] = np.array(distance_ot_reg[str(i)])

#%%
# distance_EnKF = np.load('DATA_file_keep_W2_100sims_1e5_particles_1e3_W2.npz')['distance_EnKF']
# distance_SIR = np.load('DATA_file_keep_W2_100sims_1e5_particles_1e3_W2.npz')['distance_SIR']
# distance_OT = np.load('DATA_file_keep_W2_100sims_1e5_particles_1e3_W2.npz')['distance_OT']
# distance_OT_reg = np.load('DATA_file_keep_W2_100sims_1e5_particles_1e3_W2.npz')['distance_OT_reg']

#%%
# fontsize = 16

plt.figure(figsize=(8,11))

plt.semilogy(t,distance_OT.mean(axis=1),'b--',label=r"$OT_{(\lambda=0)}$" ,lw=2)
plt.semilogy(t,distance_OT_reg.mean(axis=1),'g-.',label=r"$OT_{(\lambda=0.01)}$" ,lw=2)
plt.semilogy(t,distance_EnKF.mean(axis=1),':',color='C4',label=r"$EnKF$",lw=2.5)
plt.semilogy(t,distance_SIR.mean(axis=1),':',color='C5',label=r"$SIR$" ,lw=2.5)
# plt.xlabel(r'$time$',fontsize=fontsize)
# plt.ylabel(r'$W_2$',fontsize=fontsize)
# plt.legend(fontsize=fontsize)

plt.xlabel(r'$time$')
plt.ylabel(r'$W_2$')
plt.legend()

#%%
sim = 0 

true_particle = int(1e6)
X0_true = np.zeros((1,L,true_particle))
X0_true[0,] = 1*np.transpose(np.random.multivariate_normal(np.zeros(L),Noise[2]*Noise[2] * np.eye(L),true_particle))
X_true_dist = SIR(Y_true[sim,].reshape(1,N,dy),X0_true,L63,h,t,tau,Noise,Odeint)

#%%
for num_plot_state in range(L):
    # num_plot_state = 2
    n_bins = 50
    fontsize = 16
    if num_plot_state == 0:
        y_lim = [-30,30]
        n_ylabel = 3
    elif num_plot_state == 1:
        y_lim = [-35,35]
        n_ylabel = 3
    elif num_plot_state == 2:
         y_lim = [-20,60]
         n_ylabel = 3
    
    
    position_bins = np.linspace(y_lim[0], y_lim[1], n_bins)  # Define position bins
    
    plt.figure(figsize=(8,11))
    
    ##############################################################################
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_true_dist[0,:,num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density
    
    plt.subplot(5,1,1)
    sns.heatmap(
        density_matrix.T,
        # cmap="viridis",
        cmap= 'Purples',
        # cmap="mako",  # Choose the desired colormap
        cbar=False,  # Disable the colorbar
        robust=True
        )
    
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel),labels= np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    if num_plot_state==0:
        plt.ylabel(r'$True$')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ##############################################################################
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_EnKF[sim,:,num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density
        
    plt.subplot(5,1,2)
    sns.heatmap(
        density_matrix.T,
        # cmap="viridis",
        cmap= 'Purples',
        # cmap="mako",  # Choose the desired colormap
        cbar=False,  # Disable the colorbar
        robust=True
        )
    
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel),labels= np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    if num_plot_state==0:
        plt.ylabel(r'$EnKF$')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ##############################################################################
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_SIR[sim,:,num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density
        
    plt.subplot(5,1,3)
    sns.heatmap(
        density_matrix.T,
        # cmap="viridis",
        cmap= 'Purples',
        # cmap="mako",  # Choose the desired colormap
        cbar=False,  # Disable the colorbar
        robust=True
        )
    
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel),labels= np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    if num_plot_state==0:
        plt.ylabel(r'$SIR$')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ##############################################################################
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_OT[sim,:,num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density
        
    plt.subplot(5,1,4)
    sns.heatmap(
        density_matrix.T,
        # cmap="viridis",
        cmap= 'Purples',
        # cmap="mako",  # Choose the desired colormap
        cbar=False,  # Disable the colorbar
        robust=True
        )
    
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel),labels= np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    if num_plot_state==0:
        plt.ylabel(r'$OT~(\lambda=0)$')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ##############################################################################
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_OT_reg[sim,:,num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density
        
    plt.subplot(5,1,5)
    sns.heatmap(
        density_matrix.T,
        # cmap="viridis",
        cmap= 'Purples',
        # cmap="mako",  # Choose the desired colormap
        cbar=False,  # Disable the colorbar
        robust=True
        )
    
    ax = plt.gca()
    ax.invert_yaxis()
    # ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel),labels= np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    plt.xticks(ticks=np.linspace(0, N, 11),labels= np.round(np.linspace(0, N*tau, 11),1))
    if num_plot_state==0:
        plt.ylabel(r'$OT~(\lambda=0.01)$')
    plt.xlabel(r'$time$')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    if num_plot_state == 0:
        plt.savefig('L63_X1.png')
        plt.savefig('L63_X1.pdf')
    elif num_plot_state == 1:
        plt.savefig('L63_X2.png')
        plt.savefig('L63_X2.pdf')
    elif num_plot_state == 2:
         plt.savefig('L63_X3.png')
         plt.savefig('L63_X3.pdf')


