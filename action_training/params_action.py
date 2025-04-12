import numpy as np
import math


# Job parameters.
run_id = 'Raw_action_pahmdb51'
arch = 'largei3d'
saved_model = None
restart = False

# Dataset parameters.
num_classes = 51  # 102
num_frames = 16
fix_skip = 2
num_modes = 5
num_skips = 1
data_percentage = 1.0

# Training parameters.
batch_size = 16
v_batch_size = 16
num_workers = 4
learning_rate = 1e-4
num_epochs = 100
loss = 'ce'
val_array = [1] + [5*x for x in range(1, 8)] + [2*x for x in range(21, 35)]

temporal_distance = None

# Validation augmentation params.
hflip = [0]
cropping_facs = [0.8]
cropping_factor = 0.8
weak_aug = False
no_ar_distortion = False
aspect_ratio_aug = False

# Training augmentation params.
reso_h = 224
reso_w = 224
ori_reso_h = 240
ori_reso_w = 320
min_crop_factor_training = 0.6
temporal_align = False
