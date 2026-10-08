"""
@author: Mohammad Al-Jarrah
"""

# --- Best SMAC-Tuned Hyperparameters (see smac_tuning_Bimodal.py) ---
# Raw integer values are scaled by constants inside get_param().

# old parameter set (for reference)
# _SMAC_BEST = {
#     'ITERATION':  3,     # x512 → ITERS = 1536
#     'K_in':       5,     # x5   → inner-loop updates per outer step = 25
#     'batch_size': 12,    # x32  → BATCH_SIZE = 384
#     'lr1':        6e-4,  # learning rate for the potential network f
#     'lr2':        6e-3,  # learning rate for the transport map network T
#     'nbs1':       1,     # number of residual blocks in f
#     'nbs2':       1,     # number of residual blocks in T
#     'nns1':       12,    # x32  → NUM_NEURON_f = 384
#     'nns2':       12,    # x32  → NUM_NEURON_T = 384
#     'iter_0':     512,   # base period for cosine annealing warm restarts
# }


# SMAC config 172 (SW2 objective, d=10, N=5000, λ=0), trained for 1024 iterations
# with a single 1024-iteration cosine cycle instead of SMAC's 512 / 512.
_SMAC_BEST = {
    'ITERATION':  2,                # x512 → ITERS = 1024
    'K_in':       4,                # x5   → inner-loop updates per outer step = 20
    'batch_size': 8,                # x32  → BATCH_SIZE = 256
    'lr1':        0.0014269375086,  # learning rate for the potential network f
    'lr2':        0.0002844191263,  # learning rate for the transport map network T
    'nbs1':       1,                # number of residual blocks in f
    'nbs2':       1,                # number of residual blocks in T
    'nns1':       9,                # x32  → NUM_NEURON_f = 288
    'nns2':       12,               # x32  → NUM_NEURON_T = 384
    'iter_0':     1024,             # base period for cosine annealing warm restarts
}


# --- Parameter Builder ---

def get_param():
    """
    Return the OTF hyperparameters derived from the best SMAC-tuned configuration.

    Returns:
        tuple: (NUM_NEURON_f, NUM_NEURON_T, NUM_RESBLOCKS_f, NUM_RESBLOCKS_T,
                BATCH_SIZE, ITERS, LR_f, LR_T, K_in, ITER_0)
    """
    config = _SMAC_BEST
    NUM_NEURON_f    = int(config["nns1"] * 32)         # hidden layer width of f
    NUM_NEURON_T    = int(config["nns2"] * 32)         # hidden layer width of T
    NUM_RESBLOCKS_f = int(config["nbs1"])              # number of residual blocks in f
    NUM_RESBLOCKS_T = int(config["nbs2"])              # number of residual blocks in T
    BATCH_SIZE      = int(config["batch_size"] * 32)   # mini-batch size per iteration
    ITERS           = int(config["ITERATION"] * 512)   # total training iterations
    LR_f            = float(config["lr1"])             # learning rate for f
    LR_T            = float(config["lr2"])             # learning rate for T
    K_in            = int(config["K_in"] * 5)          # inner-loop updates per outer step
    ITER_0          = int(config["iter_0"])            # cosine annealing restart period
    return NUM_NEURON_f, NUM_NEURON_T, NUM_RESBLOCKS_f, NUM_RESBLOCKS_T, BATCH_SIZE, ITERS, LR_f, LR_T, K_in, ITER_0
