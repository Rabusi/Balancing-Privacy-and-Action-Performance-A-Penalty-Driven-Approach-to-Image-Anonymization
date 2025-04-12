import numpy as np
import math
import os.path


# Job parameters.
run_id = 'Anonymization_balanced_0.5'
arch_ft = 'largei3d'
arch_fa = 'unet++'
arch_fb = 'r50'
saved_model_fa = os.path.join('', 'model_weights/fa_recon.pth')
saved_model_ft = None
saved_model_fb = os.path.join('','fb_ssl.pth')

# Dataset parameters UCF101.
num_classes = 102
num_frames = 16
fix_skip = 2
num_modes = 5
num_skips = 1
data_percentage = 1.0

# Dataset parameters HMDB51.
# num_classes = 51
# num_frames = 16
# fix_skip = 2
# num_modes = 5
# num_skips = 1
# data_percentage = 1.0


# Number of VISPR privacy attributes.
num_pa = 7
data_percentage_vispr = 1.0

# budget penalty parameter
B = 0.50
lambda_budget = 1.0
budget_penalty_loss = 'rms_loss'

# Training parameters.
batch_size = 8
batch_size_vispr = 8
v_batch_size = 8
num_workers = 4
learning_rate = 1e-4
num_epochs = 100
loss = 'ce'
opt_type = 'adam' # 'sgd' # 'adamw'
ft_dropout = 0

# Anonymization training parameters.
# Scaled lr per model.
learning_rate_fa = 0.4*learning_rate
learning_rate_fb = 1.0*learning_rate
learning_rate_ft = 1.0*learning_rate
ft_loss_weight = 1.0 # 0.7
fb_loss_weight = 1.0
weight_inv = 0.0
triplet_loss_margin = 1
temporal_distance = None

# Validation augmentation params.
hflip = [0]
cropping_facs = [0.8]
weak_aug = False
no_ar_distortion = False
aspect_ratio_aug = False

# Training augmentation params.
reso_h = 224
reso_w = 224
min_crop_factor_training = 0.6
temporal_align = False

