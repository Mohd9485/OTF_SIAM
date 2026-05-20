"""
@author: Mohammad Al-Jarrah
"""

# Best hyperparameters selected after SMAC tuning (see smac_tuning_L96.py).
# Raw integer values are scaled by constants in get_params().
_SMAC_BEST = {
    'batch_size':       3,     # x32  → BATCH_SIZE = 96
    'inner_iterations': 2,     # inner T-update steps per outer iteration
    'iteration':        1,     # x512 → ITERATION  = 512
    'lr_T':             5e-4,  # learning rate for the transport map network T
    'lr_f':             5e-4,  # learning rate for the potential network f
    'num_neuron_T':     4,     # x32  → NUM_NEURON_T = 128
    'num_neuron_f':     4,     # x32  → NUM_NEURON_f = 128
}


def get_params(L, dy):
    """
    Build the OTF parameters dict from the best SMAC-tuned hyperparameters.

    Parameters
    ----------
    L  : int — state space dimension
    dy : int — observation space dimension

    Returns
    -------
    dict — hyperparameter dictionary consumed by OTF()
    """
    return {
        'normalization':          'None',
        'INPUT_DIM':              [L, dy],
        'NUM_NEURON':             [_SMAC_BEST['num_neuron_f'] * 32,
                                   _SMAC_BEST['num_neuron_T'] * 32],
        'BATCH_SIZE':             _SMAC_BEST['batch_size'] * 32,
        'LearningRate':           [_SMAC_BEST['lr_f'], _SMAC_BEST['lr_T']],
        'ITERATION':              _SMAC_BEST['iteration'] * 512,
        'Final_Number_ITERATION': 64,
        'inner_iterations':       _SMAC_BEST['inner_iterations'],
    }
