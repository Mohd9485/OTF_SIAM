"""
@author: Mohammad Al-Jarrah
"""

# --- Hyperparameters ---

# Raw tuner values from SMAC (smac_tuning_L63.py) — multiplied by scaling constants in get_params()
_SMAC_BEST = {
    'num_neuron': 1,  # x32  → NUM_NEURON = 32
    'batch_size': 4,  # x32  → BATCH_SIZE = 128
    'lr'        : 2e-3,  # learning rate
    'iteration' : 1,  # x512 → ITERATION  = 512
}


# --- Parameter Builder ---

def get_params(L, dy):
    """
    Build the OTF parameter dict using the best SMAC-tuned hyperparameters.

    Args:
        L  (int): lag / history length used as the first input dimension
        dy (int): observation dimension used as the second input dimension

    Returns:
        dict: training configuration for the OTF model
    """
    return {
        'normalization':          'None',                          # no input normalization applied
        'INPUT_DIM':              [L, dy],                         # model input shape: [lag, obs_dim]
        'NUM_NEURON':             _SMAC_BEST['num_neuron'] * 32,   # hidden layer width
        'BATCH_SIZE':             _SMAC_BEST['batch_size'] * 32,   # mini-batch size
        'LearningRate':           _SMAC_BEST['lr'],                # optimizer step size
        'ITERATION':              _SMAC_BEST['iteration'] * 512,   # total training iterations
        'Final_Number_ITERATION': 64,                              # iterations for the final refinement stage
    }
