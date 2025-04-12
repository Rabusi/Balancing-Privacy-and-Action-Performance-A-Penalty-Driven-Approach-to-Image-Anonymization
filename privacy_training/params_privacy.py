import numpy as np


# Job parameters.
run_id = 'privacy_vispr1_fa_budget_penalty'
# saved_model = None
saved_model = './home/Model_Weights/model_weights/model.pth'
# anon = False
anon = True
arch = 'r152'

# Number of VISPR privacy attributes.
num_pa = 7
data_percentage = 1.0

# Training parameters.
batch_size = 8
v_batch_size = 8
num_workers = 0
learning_rate = 1e-3
num_epochs = 100
warmup_array = list(np.linspace(0.01, 1, 5) + 1e-9)
warmup = len(warmup_array)
lr_reduce_factor = 5
lr_patience = 0

# Validation augmentation params.
hflip = [0]
cropping_fac1 = [0.8]

# Training augmentation params.
reso_h = 224
reso_w = 224
