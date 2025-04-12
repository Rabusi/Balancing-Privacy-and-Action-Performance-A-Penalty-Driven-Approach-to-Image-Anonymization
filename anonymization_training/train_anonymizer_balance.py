import argparse
import importlib
import itertools
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorboardX import SummaryWriter
import time
import torch
import torch.optim as optim
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import sys
# sys.path.insert(0, '..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aux_code import config as cfg
from aux_code.model_loaders import load_ft_model, load_fa_model, load_fb_model
from aux_code.VPUCF_dl import *
from aux_code.ucf101_dl import *
# from aux_code.hmdb51_dl import *
# from aux_code.VPHMDB_dl import *
from aux_code.nt_xent_original import NTXentLoss
from aux_code.vispr_dl_edit import  vispr_dataset, vispr_ssl_dataset

import params_anonymization as  params

torch.backends.cudnn.benchmark = True

# Training epoch.
def train_epoch(epoch, dataloader_vispr, dataloader_video, ft_model, fa_model, fb_model, criterion_ft, 
                criterion_temporal_ft, optimizer_fa, optimizer_fb, optimizer_ft, writer, use_cuda, learning_rate_fa, learning_rate_fb, 
                learning_rate_ft, device_name, params):
    print(f'Train at epoch {epoch}')
    for param_group_fa in optimizer_fa.param_groups:
        param_group_fa['lr'] = learning_rate_fa
    for param_group_fb in optimizer_fb.param_groups:
        param_group_fb['lr'] = learning_rate_fb
    for param_group_ft in optimizer_ft.param_groups:
        param_group_ft['lr'] = learning_rate_ft
        
    writer.add_scalar('Learning Rate Fa', learning_rate_fa, epoch)
    writer.add_scalar('Learning Rate Fb', learning_rate_fb, epoch)
    writer.add_scalar('Learning Rate Ft', learning_rate_ft, epoch)  
    print(f'Learning rate of fa is: {param_group_fa["lr"]}')
    print(f'Learning rate of fb is: {param_group_fb["lr"]}')
    print(f'Learning rate of ft is: {param_group_ft["lr"]}')
    
    losses_fa, losses_fb, losses_ft, losses_temporal, losses_budget, B_step1 = [], [], [], [], [], []

    torch.autograd.set_detect_anomaly(True)

    step = 1
    for i, (data1, data2) in enumerate(zip(dataloader_vispr, dataloader_video)):
        inputs_video, labels_video = data2[0:2]
        
        # Ensure that labels are valid (0 to 50 for 51 classes for HMDB dataset)
        # assert (labels_video >= 0).all() and (labels_video < 51).all(), "Invalid labels detected."

        # Check if inputs_video is None
        if inputs_video is None:
            print(f"Skipping batch {i + 1} because inputs_video is None")
            continue

        # Check for NaN or Inf in inputs_video
        if isinstance(inputs_video, torch.Tensor):
            if torch.isnan(inputs_video).any() or torch.isinf(inputs_video).any():
                print(f"Skipping batch {i + 1} because inputs_video contains NaN or Inf")
                continue
            if inputs_video.numel() == 0:
                print(f"Skipping batch {i + 1} because inputs_video is empty")
                continue

        inputs_vispr = [data1[ii] for ii in range(2)]
        if any(x is None for x in inputs_vispr):
            print(f'Skipping the batch {i+1}: inputs_vispr is None in step_{step}')
            
        # Check if any tensor in inputs_vispr has NaN or Inf values
        for input_tensor in inputs_vispr:
            if isinstance(input_tensor, torch.Tensor):
                if torch.isnan(input_tensor).any():
                    print(f'Skipping batch {i+1}: inputs_vispr contains NaN values')
                    continue
                if torch.isinf(input_tensor).any():
                    print(f'Skipping batch {i+1}: inputs_vispr contains Inf values')
                    continue
                
        inputs_video = inputs_video.permute(0,2,1,3,4)

        if use_cuda:
            inputs_vispr = [inputs_vispr[ii].to(device=torch.device(device_name), non_blocking=True) for ii in range(2)]
            # Do NOT use VISPR privacy labels in training.
            # labels_vispr = torch.from_numpy(np.asarray(data1[2])).float().cuda()
            inputs_video = inputs_video.to(device=torch.device(device_name), non_blocking=True)
            labels_video = torch.from_numpy(np.asarray(labels_video)).type(torch.LongTensor).to(device=torch.device(device_name), non_blocking=True)

        optimizer_fa.zero_grad()
        optimizer_fb.zero_grad()
        optimizer_ft.zero_grad()

        # Step-1: Update fa.
        if step == 1:
            # Train fa model, not ft and fb.
            fa_model.train()
            ft_model.eval()
            fb_model.eval()

            # Autocast automatic mixed precision.
            with autocast():
                # Get anonymous reconstruction from fa, input to fb.
                output_fa = [fa_model(inputs_vispr[ii]) for ii in range(2)]
                output_fb = [fb_model(output_fa[ii]) for ii in range(2)]
                
                # Check for NaN values in output_fb before loss computation
                if torch.isnan(output_fb[0]).any():
                    print(f'skip output_fb[0]: nan value in step_{step}')
                    continue
                if torch.isnan(output_fb[1]).any():
                    print(f'skip output_fb[1]: nan value in step_{step}')
                    continue
                if torch.isinf(output_fb[0]).any():
                    print(f'skip output_fb[0]: inf value in step_{step}')
                    continue
                if torch.isinf(output_fb[1]).any():
                    print(f'skip output_fb[0]: nan value in step_{step}')
                    continue
                
                # Contrastive loss function for SSL.
                con_loss_criterion = NTXentLoss(device='cuda', batch_size=output_fb[0].shape[0], temperature=0.1, use_cosine_similarity=False)
                # Compute losses.
                loss_fb = con_loss_criterion(output_fb[0], output_fb[1])
                if torch.isnan(loss_fb).any() or torch.isinf(loss_fb).any():
                    print("loss_fb contains NaN or Inf. Skipping batch.")
                    continue

                # Split original shape up.
                inputs_video_ori = inputs_video
                ori_bs, ori_t, ori_c, ori_h, ori_w = inputs_video.shape
                # Reshape video input.
                inputs_video = inputs_video.reshape(-1, inputs_video.shape[1], inputs_video.shape[3], inputs_video.shape[4])

                # Get anonymous reconstruction from fa, input to ft.
                anon_input = fa_model(inputs_video).reshape(ori_bs, ori_t, ori_c, ori_h, ori_w)
                
                # Penalty on video data
                if params.budget_penalty_loss == 'rms_loss':
                    def compute_rms(tensor):
                        return torch.sqrt(torch.mean(tensor ** 2))
                    delta_I = torch.tensor([compute_rms(inputs_video_ori - anon_input)]).cuda()
                    delta_I_minus_B = delta_I - params.B
                    loss_budget = max(0, torch.max(delta_I_minus_B).item())

                if loss_budget > 0:
                    print(f"L_budget is :({loss_budget}). Penalty will be applied")
                

                inputs1 = anon_input
                output, feat1 = ft_model(inputs1)
                if params.loss == 'ce':
                    # Compute loss.
                    loss_ft = criterion_ft(output, labels_video)                    

                # Combine losses into single fa loss.
                loss_fa = -params.fb_loss_weight*loss_fb + params.ft_loss_weight*loss_ft + params.lambda_budget*loss_budget
            
            losses_fa.append(loss_fa.item())
            losses_budget.append(loss_budget)
            
            loss_fa.backward()
            B_step1.append(params.B.item())   # append the value of B from each batch
            
            # Update fa
            optimizer_fa.step()

            # Set to step 2 to update other networks.
            step = 2
            if i % 100 == 0 or i % 100 == 1:
                print(f'Training Epoch {epoch}, Batch {i}, loss_fa: {np.mean(losses_fa) :.5f}, loss_budget: {np.mean(losses_budget) :.5f}', flush = True)
            # Skip step 2 for this batch, go to next batch.
            continue
        
        # Step-2: Update ft and fb.
        if step == 2:
            fa_model.eval()
            fb_model.train()
            ft_model.train()

            # Run inputs through fa_model.
            with torch.no_grad():
                # Split original shape up.
                ori_bs, ori_t, ori_c, ori_h, ori_w = inputs_video.shape
                # Reshape video input.
                inputs_video = inputs_video.reshape(-1, inputs_video.shape[1], inputs_video.shape[3], inputs_video.shape[4])
                # input1 = [fa_model(inputs_vispr[ii]) for ii in range(2)]
                input1 = [torch.clamp(fa_model(inputs_vispr[ii]), min=-1e5, max=1e5) for ii in range(2)]
                input2 = fa_model(inputs_video).reshape(ori_bs, ori_t, ori_c, ori_h, ori_w)
                
                # Check if any element in input1 is None
                if any(x is None for x in input1):
                    print(f'Skipping batch {i+1}: fa_model(inputs_vispr) produced None in step_{step}')
                    continue
                
                skip_batch = False

                # Check if any tensor in input1 contains NaN or Inf values
                for input_tensor in input1:
                    if isinstance(input_tensor, torch.Tensor):
                        if torch.isnan(input_tensor).any():
                            print(f'Skipping batch {i+1}: fa_model(inputs_vispr) contains NaN values in step_{step}')
                            skip_batch = True
                            break
                        if torch.isinf(input_tensor).any():
                            print(f'Skipping batch {i+1}: fa_model(inputs_vispr) contains Inf values in step_{step}')
                            skip_batch = True
                            break

                if skip_batch:
                    continue

            # Autocast automatic mixed precision.
            with autocast():
                # Get anonymous reconstruction from fa, input to fb.
                output1 = [fb_model(x) for x in input1]
                
                # Check for NaN values in output1 before loss computation
                if torch.isnan(output1[0]).any():
                    print(f'skip output1[0]: nan value in step_{step}')
                    continue
                if torch.isnan(output1[1]).any():
                    print(f'skip output1[1]: nan value in step_{step}')
                    continue
                if torch.isinf(output1[0]).any():
                    print(f'skip output1[0]: inf value in step_{step}')
                    continue
                if torch.isinf(output1[1]).any():
                    print(f'skip output1[0]: nan value in step_{step}')
                    continue

                # Contrastive loss function for SSL.
                con_loss_criterion = NTXentLoss(device='cuda', batch_size=output1[0].shape[0], temperature=0.1, use_cosine_similarity=False)
                # Compute losses.
                loss_fb = con_loss_criterion(output1[0], output1[1])
                if torch.isnan(loss_fb).any() or torch.isinf(loss_fb).any():
                    print("loss_fb contains NaN or Inf. Skipping batch.")
                    continue

                # Get anonymous reconstruction from fa, input to ft.
                inputs1 = input2
                output2, feat1 = ft_model(inputs1)
                if params.loss == 'ce':
                    # Compute loss.
                    loss_ft = criterion_ft(output2, labels_video)

            losses_ft.append(float(loss_ft.item()))
            losses_fb.append(float(loss_fb.item()))

            loss_fb.backward()
            loss_ft.backward()

            optimizer_fb.step()
            optimizer_ft.step()
            # Set to step 2 to update other network.
            step = 1
            if i % 100 == 0 or i % 100 == 1:
                print(f'Training Epoch {epoch}, Batch {i}, loss_fb: {np.mean(losses_fb) :.5f}, loss_ft: {np.mean(losses_ft) :.5f}', flush = True)
            continue
        

    # Compute current mean losses
    loss_fa = np.mean(losses_fa)
    # loss_budget = np.mean(losses_budget)
    loss_ft = np.mean(losses_ft)

    print(f"Training Epoch: {epoch}" , 
        f"loss_fa: {np.mean(losses_fa):.4f}, "
        f"loss_fb: {np.mean(losses_fb):.4f}, "
        f"loss_ft: {np.mean(losses_ft):.4f}"
        )
    
    writer.add_scalar('Training loss_fa', np.mean(losses_fa), epoch)
    writer.add_scalar('Training loss_fb', np.mean(losses_fb), epoch)
    writer.add_scalar('Training loss_ft', np.mean(losses_ft), epoch)

    del loss_fb, inputs_vispr, inputs_video, inputs1, output1, output2, anon_input, input1, input2

    return fa_model, fb_model, ft_model


# Validation epoch.
def val_epoch_video(epoch, mode, cropping_fac, pred_dict, label_dict, data_loader, ft_model, fa_model, criterion, criterion_temporal, use_cuda, device_name, params):
    print(f'Validation at epoch {epoch}.')
    
    # Set models to eval.
    ft_model.eval()
    fa_model.eval()

    losses = []
    predictions, ground_truth = [], []
    vid_paths = []

    for i, (inputs, label, vid_path, _) in enumerate(data_loader):
        if vid_path == None or inputs.shape[0] != params.v_batch_size:
            continue
        #print(i)
        vid_paths.extend(vid_path)
        ground_truth.extend(label)
        inputs = inputs.permute(0,2,1,3,4)
        
        if use_cuda:
            inputs = inputs.to(device=torch.device(device_name), non_blocking=True)
            label = torch.from_numpy(np.asarray(label)).type(torch.LongTensor).to(device=torch.device(device_name), non_blocking=True)
            # label = label -1
            # assert torch.all((label >= 0) & (label < 51)), f"Label out of range after transformation: {label}"

        with torch.no_grad():
            # Split original shape up.
            ori_bs, ori_t, ori_c, ori_h, ori_w = inputs.shape
            # Reshape video input.
            inputs = inputs.reshape(-1, inputs.shape[1], inputs.shape[3], inputs.shape[4])
            anon_input = fa_model(inputs).reshape(ori_bs, ori_t, ori_c, ori_h, ori_w)

            inputs1 = anon_input

            output, feat1 = ft_model(inputs1)

            if params.loss == 'ce':
                # Compute loss.
                loss = criterion(output, label)

        losses.append(loss.item())

        predictions.extend(nn.functional.softmax(output, dim=1).cpu().data.numpy())

        if i % 200 == 0:
            print(f'Validation Epoch {epoch}, Batch {i}, Loss : {np.mean(losses)}', flush=True)
        
    del inputs, output, label, loss, anon_input

    ground_truth = np.asarray(ground_truth)
    pred_array = np.flip(np.argsort(predictions, axis=1), axis=1) 
    c_pred = pred_array[:, 0] 

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])
        else:
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    correct_count = np.sum(c_pred==ground_truth)
    accuracy = float(correct_count)/len(c_pred)

    print(f'Epoch {epoch}, mode {mode}, cropping_fac {cropping_fac} - Accuracy: {accuracy*100:.3f}%')

    return pred_dict, label_dict, accuracy, np.mean(losses)


# Visualize anonymized reconstruction on validation epoch.
def val_visualization_fa_vispr(save_dir, epoch, validation_dataloader, fa_model):
    #with torch.inference_mode():
    with torch.no_grad():
        for inputs, _, _ in validation_dataloader:
            if len(inputs.shape) == 1:
                continue
            inputs = inputs.cuda()
            image_full_name = os.path.join(save_dir, f'combined_epoch{epoch}.png')
            outputs = fa_model(inputs)
            vis_image = torch.cat([inputs, outputs], dim=0)
            save_image(vis_image, image_full_name, padding=5, nrow=int(inputs.shape[0]))
            return


# Main code. 
def train_classifier(params, devices):
    # print relevant parameters.
    for k, v in params.__dict__.items():
        if '__' not in k:
            print(f'{k} : {v}')
    # Empty cuda cache.
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    writer = SummaryWriter(os.path.join(cfg.logs, str(params.run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Load pretrained reconstruction fa_model.
    fa_model = load_fa_model(arch=params.arch_fa, saved_model_file=params.saved_model_fa)
    # Load in pretrained ft_model. 
    ft_model = load_ft_model(arch=params.arch_ft, saved_model_file=params.saved_model_ft, num_classes=params.num_classes, kin_pretrained=True if params.saved_model_ft is None else False)
    # Load in pretrained fb_model.
    fb_model = load_fb_model(arch=params.arch_fb, saved_model_file=params.saved_model_fb, ssl=True)
    
    # Select optimizer.
    if params.opt_type == 'adam':
        optimizer_fa = torch.optim.Adam(fa_model.parameters(), lr=params.learning_rate_fa)
        optimizer_fb = torch.optim.Adam(fb_model.parameters(), lr=params.learning_rate_fb)
        optimizer_ft = torch.optim.Adam(ft_model.parameters(), lr=params.learning_rate_ft)
    elif params.opt_type == 'adamw':
        optimizer_fa = torch.optim.AdamW(fa_model.parameters(), lr=params.learning_rate_fa, weight_decay=params.weight_decay)
        optimizer_fb = torch.optim.AdamW(fb_model.parameters(), lr=params.learning_rate_fb, weight_decay=params.weight_decay)
        optimizer_ft = torch.optim.AdamW(ft_model.parameters(), lr=params.learning_rate_ft, weight_decay=params.weight_decay)
    elif params.opt_type == 'sgd':
        optimizer_fa = torch.optim.SGD(fa_model.parameters(), lr=params.learning_rate_fa, momentum=params.momentum, weight_decay=params.weight_decay)
        optimizer_fb = torch.optim.SGD(fb_model.parameters(), lr=params.learning_rate_fb, momentum=params.momentum, weight_decay=params.weight_decay)
        optimizer_ft = torch.optim.SGD(ft_model.parameters(), lr=params.learning_rate_ft, momentum=params.momentum, weight_decay=params.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {params.opt_type} not yet implemented.')
    
    learning_rate_fa = params.learning_rate_fa
    learning_rate_fb = params.learning_rate_fb
    learning_rate_ft = params.learning_rate_ft

    # Init loss functions.
    if params.loss == 'con':
        criterion_ft = None
    elif params.loss == 'ce':
        criterion_ft = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f'Loss function {params.loss} not yet implemented.')

    if params.temporal_loss == 'trip':
        criterion_temporal_ft = nn.TripletMarginLoss(margin=params.triplet_loss_margin)
    else:
        criterion_temporal_ft = None

    device_name = f'cuda:{devices[0]}'
    print(f'Device name is {device_name}')
    if len(devices) > 1:
        print(f'Multiple GPUS found!')
        ft_model = nn.DataParallel(ft_model, device_ids=devices)
        fa_model = nn.DataParallel(fa_model, device_ids=devices)
        fb_model = nn.DataParallel(fb_model, device_ids=devices)
        ft_model.cuda()
        fa_model.cuda()
        fb_model.cuda()
        criterion_ft.cuda()
    else:
        print('Only 1 GPU is available')
        ft_model.to(device=torch.device(device_name))
        fa_model.to(device=torch.device(device_name))
        fb_model.to(device=torch.device(device_name))
        criterion_ft.to(device=torch.device(device_name))
            
    
    # Check if a checkpoint exists
    checkpoint_path = os.path.join('')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        fa_model.load_state_dict(checkpoint['fa_model_state_dict'])
        fb_model.load_state_dict(checkpoint['fb_model_state_dict'])
        ft_model.load_state_dict(checkpoint['ft_model_state_dict'])
        epoch1 = checkpoint['epoch']
        optimizer_fa.load_state_dict(checkpoint['optimizer_fa'])
        optimizer_fb.load_state_dict(checkpoint['optimizer_fb'])
        optimizer_ft.load_state_dict(checkpoint['optimizer_ft'])
        print(f"Resuming training from epoch {epoch1}")
    else:
        print("No checkpoint found, starting from scratch.")
        epoch1 = 1
        

    train_dataset_vispr = vispr_ssl_dataset(data_split='train', shuffle=True, data_percentage=params.data_percentage_vispr)
    train_dataloader_vispr = DataLoader(
        train_dataset_vispr,
        batch_size=params.batch_size_vispr,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=True,drop_last = True)
    print(f'VISPR Train dataset length: {len(train_dataset_vispr)}')
    print(f'VISPR Train dataset steps per epoch: {len(train_dataset_vispr)/params.batch_size_vispr}')

    modes = list(range(params.num_modes))
    cropping_facs = params.cropping_facs
    modes, cropping_fac = list(zip(*itertools.product(modes, cropping_facs)))

    val_array = [1, 5, 10, 12, 15, 20, 25, 30, 35] + [40 + x*2 for x in range(30)]

    print(f'Num modes {len(modes)}')
    print(f'Cropping fac {cropping_facs}')
    print(f'Base learning rate {params.learning_rate}')

    accuracy = 0

    for epoch in range(epoch1, params.num_epochs + 1):
        print(f'Epoch {epoch} started')
        start = time.time()
        torch.multiprocessing.set_sharing_strategy('file_system')

        action_name = cfg.hmdb51_class_mapping
        train_dataset_video = single_train_dataloader(params=params, shuffle=True, data_percentage=params.data_percentage)
        # train_dataset_video = single_train_dataloader_vpucf(params=params, shuffle=True, data_percentage=params.data_percentage)
        # train_dataset_video = single_train_dataloader_hmdb(params=params, shuffle=True, data_percentage=params.data_percentage, action_name=action_name)
        # train_dataset_video = single_train_dataloader_vphmdb(params=params, shuffle=True, data_percentage=params.data_percentage, action_name=action_name)

        if epoch == epoch1:
            print(f'Video Train dataset length: {len(train_dataset_video)}')
            print(f'Video Train dataset steps per epoch: {len(train_dataset_video)/params.batch_size}')

        train_dataloader_video = DataLoader(
            train_dataset_video, 
            shuffle=False,
            batch_size=params.batch_size, 
            num_workers=params.num_workers,
            collate_fn=collate_fn_train,
            pin_memory=True, drop_last = True)

        fa_model, fb_model, ft_model = train_epoch(epoch, train_dataloader_vispr, train_dataloader_video, ft_model, 
                                                                                          fa_model, fb_model,                                                                                     criterion_ft,criterion_temporal_ft, optimizer_fa, 
                                                                                          optimizer_fb, optimizer_ft, writer, use_cuda, learning_rate_fa, 
                                                                                          learning_rate_fb, learning_rate_ft, device_name, params)
        
        validation_dataset_vispr = vispr_dataset(data_split='test', shuffle=True, data_percentage=1.0)
        validation_dataloader = DataLoader(validation_dataset_vispr, batch_size=32, shuffle=False, num_workers=params.num_workers, drop_last=True)
        val_visualization_fa_vispr(save_dir, epoch, validation_dataloader, fa_model)
        
        # Validation epoch.
        if epoch in val_array:
            pred_dict, label_dict = {}, {}
            val_losses = []

            for val_iter, mode in enumerate(modes):
                for cropping_fac in cropping_facs:
                    validation_dataset_video = single_val_dataloader(params=params, shuffle=True, data_percentage=1.0, mode=mode)
                    # validation_dataset_video = single_val_dataloader_vpucf(params=params, shuffle=True, data_percentage=1.0, mode=mode)
                    # validation_dataset_video = single_val_dataloader_hmdb(params=params, shuffle=True, data_percentage=1.0, mode=mode, action_name=action_name)
                    # validation_dataset_video = single_val_dataloader_vphmdb(params=params, shuffle=True, data_percentage=1.0, mode=mode, action_name=action_name)
                    
                    validation_dataloader_video = DataLoader(
                        validation_dataset_video, 
                        batch_size=params.v_batch_size, 
                        shuffle=False, num_workers=params.num_workers, 
                        collate_fn=collate_fn_val, drop_last = True, 
                        pin_memory=True)
                    
                    if val_iter == 0:
                        print(f'Video Validation dataset length: {len(validation_dataset_video)}')
                        print(f'Video Validation dataset steps per epoch: {len(validation_dataset_video)/params.v_batch_size}')

                    pred_dict, label_dict, accuracy, loss = val_epoch_video(epoch, mode, cropping_fac, pred_dict, label_dict, validation_dataloader_video, ft_model, fa_model, criterion_ft, criterion_temporal_ft, use_cuda, device_name, params)
                    val_losses.append(loss)

                    predictions = np.zeros((len(list(pred_dict.keys())), params.num_classes))
                    ground_truth = []
                    for entry, key in enumerate(pred_dict.keys()):
                        predictions[entry] = np.mean(pred_dict[key], axis=0)

                    for key in label_dict.keys():
                        ground_truth.append(label_dict[key])

                    pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)  # Prediction with the most confidence is the first element here.
                    c_pred = pred_array[:, 0]

                    correct_count = np.sum(c_pred==ground_truth)
                    accuracy_all = float(correct_count)/len(c_pred)
                    print(f'Running Avg Accuracy for epoch {epoch}, mode {modes[val_iter]}, cropping_fac {cropping_fac}is {accuracy_all*100:.3f}%')

            val_loss = np.mean(val_losses)
            predictions = np.zeros((len(list(pred_dict.keys())), params.num_classes))
            ground_truth = []

            for entry, key in enumerate(pred_dict.keys()):
                predictions[entry] = np.mean(pred_dict[key], axis=0)

            for key in label_dict.keys():
                ground_truth.append(label_dict[key])

            pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)  # Prediction with the most confidence is the first element here.
            c_pred = pred_array[:,0]

            correct_count = np.sum(c_pred==ground_truth)
            accuracy = float(correct_count)/len(c_pred)
            print(f'Val loss for epoch {epoch} is {val_loss}')
            print(f'Correct Count is {correct_count} out of {len(c_pred)}')
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Validation Accuracy', accuracy, epoch)
            print(f'Overall Ft accuracy for epoch {epoch} is {accuracy*100:.3f}%')

            if accuracy > 0.6:
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} has above 60% acc for {params.run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, f'model_{epoch}_bestAcc_{str(accuracy)[:6]}_B_opt_{B.item():.4f}.pth')
                states = {
                    'epoch': epoch + 1,
                    'fa_model_state_dict': fa_model.state_dict(),
                    'fb_model_state_dict': fb_model.state_dict(),
                    'ft_model_state_dict': ft_model.state_dict(),
                    'optimizer_fa': optimizer_fa.state_dict(),
                    'optimizer_fb': optimizer_fb.state_dict(),
                    'optimizer_ft': optimizer_ft.state_dict()
                }
                torch.save(states, save_file_path)

        # We will save optimizer weights for each temp model, not all saved models to reduce the storage.
        save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
        save_file_path = os.path.join(save_dir, 'model_temp.pth')
        states = {
            'epoch': epoch + 1,
            'fa_model_state_dict': fa_model.state_dict(),
            'fb_model_state_dict': fb_model.state_dict(),
            'ft_model_state_dict': ft_model.state_dict(),
            'optimizer_fa': optimizer_fa.state_dict(),
            'optimizer_fb': optimizer_fb.state_dict(),
            'optimizer_ft': optimizer_ft.state_dict()
        }
        torch.save(states, save_file_path)

        # Save every 3 to save space.
        if epoch % 3 == 0:
            save_file_path = os.path.join(save_dir, f'model_{epoch}_B_opt_{B.item():.4f}.pth')
            states = {
                'epoch': epoch + 1,
                'fa_model_state_dict': fa_model.state_dict(),
                'fb_model_state_dict': fb_model.state_dict(),
                'ft_model_state_dict': ft_model.state_dict(),
                'optimizer_fa': optimizer_fa.state_dict(),
                'optimizer_fb': optimizer_fb.state_dict(),
                'optimizer_ft': optimizer_ft.state_dict()
            }
            torch.save(states, save_file_path)

        taken = time.time() - start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--params", dest='params', type=str, required=False, default='params_anonymization.py', help='params')
    parser.add_argument("--devices", dest='devices', action='append', type=int, required=False, default=None, help='devices should be a list')

    args = parser.parse_args()
    if os.path.exists(args.params):
        params = importlib.import_module(args.params.replace('.py', ''))
        print(f'{args.params} is loaded as parameter file.')
    else:
        print(f'{args.params} does not exist, change to valid filename.')

    if args.devices is None:
        args.devices = list(range(torch.cuda.device_count()))

    train_classifier(params, args.devices)
