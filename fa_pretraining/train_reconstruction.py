import argparse
import itertools
import numpy as np
import os
from tensorboardX import SummaryWriter
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import traceback
import sys
import logging

from reconstruction_dl import reconstruction_dataset
import parameters as params

from aux_code import config as cfg
from aux_code.model_loaders import load_fa_model

# Find optimal algorithms for the hardware.
torch.backends.cudnn.benchmark = True


log_file_path = os.path.join(cfg.training_logs , 'training_log.txt')
if not os.path.exists(cfg.training_logs  ):
    os.makedirs(cfg.training_logs )

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(log_file_path),
    logging.StreamHandler()
])


def train_epoch(epoch, data_loader, fa_model, criterion, optimizer, writer, use_cuda, lr):
    logging.info(f'Train epoch {epoch}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        writer.add_scalar('Learning_Rate', lr, epoch)  
        logging.info(f'Learning rate is: {param_group["lr"]}')
  
    losses = []

    fa_model.train()

    # Iterate through data loader.
    for i, (inputs, _) in enumerate(data_loader):
        if use_cuda:
            inputs = inputs.cuda()

        # Don't compute gradients on optimizer.
        optimizer.zero_grad()
        
        output = fa_model(inputs)
        loss = criterion(output, inputs)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 500 == 0: 
            logging.info(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}')
        
    logging.info(f'Training Epoch: {epoch}, Loss: {np.mean(losses):.4f}')
    writer.add_scalar('Training_Loss', np.mean(losses), epoch)

    del loss, inputs, output

    return fa_model, np.mean(losses)


def validation_epoch(run_id, epoch, data_loader, fa_model,  criterion, writer, use_cuda):
    fa_model.eval()

    losses = []
    vid_paths = []
    save_dir = os.path.join(cfg.saved_models_dir, run_id)

    # Iterate through dataloader.
    for i, (inputs, vid_path) in enumerate(data_loader):
        vid_paths.extend(vid_path)
        if len(inputs.shape) != 1:
            if use_cuda:
                inputs = inputs.cuda()
            
            with torch.no_grad():    
                output = fa_model(inputs)
                loss = criterion(output, inputs)
                losses.append(loss.item())
        else:
            logging.info(f'Input shape failure: {inputs.shape}')

        if i % 300 == 0: 
            logging.info(f'--------Validation Epoch {epoch}, Batch {i}, Loss: {np.mean(losses):.5f}--------')
            # Visualize images.
            vis_image = torch.cat([inputs[:5], output[:5]], dim=0)
            save_image(vis_image, os.path.join(save_dir, f'visualize_epoch_{epoch}_batch_{i}.png'), padding=5, nrow=5)

    del inputs, output, loss 
    
    logging.info(f'--------Validation Epoch: {epoch}, Loss: {np.mean(losses):4f}--------')
    writer.add_scalar('Validation_Loss', np.mean(losses), epoch)
    return np.mean(losses)
    

def train_classifier(run_id, saved_model):
    use_cuda = torch.cuda.is_available()
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    # Create save directory if it does not exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize fa_model.
    fa_model = load_fa_model(arch='unet++', saved_model_file=saved_model)

    epoch1 = 1
    lr = params.learning_rate

    # L1 reconstruction loss. 
    criterion = nn.L1Loss()

    if torch.cuda.device_count() > 1:
        logging.info('Multiple GPUS found!')
        fa_model = nn.DataParallel(fa_model)
        criterion.cuda()
        fa_model.cuda()
    else:
        logging.info('Only 1 GPU is available')
        criterion.cuda()
        fa_model.cuda()

    optimizer = optim.Adam(fa_model.parameters(), lr=params.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.80)
    train_dataset = reconstruction_dataset(data_split='train', shuffle=False, ucf101_percentage=0.01, data_percentage=1.0)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)

    logging.info(f'Train dataset length: {len(train_dataset)}')
    logging.info(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')

    # Array of epochs to run validation (skip some to save time).
    val_array = [1, 3, 5, 10, 12, 15, 20, 25, 30, 35, 40, 45] + [50 + x for x in range(100)]

    logging.info(f'Base learning rate: {params.learning_rate}')
    logging.info(f'Scheduler patience: {params.lr_patience}')
    logging.info(f'Scheduler drop: {params.scheduled_drop}')

    lr_flag = 0
    lr_counter = 0
    train_loss_best = 10000
    val_loss_best = 10000

    for epoch in range(epoch1, params.num_epochs + 1):
        if epoch < params.warmup and lr_flag == 0:
            lr = params.warmup_array[epoch]*params.learning_rate

        logging.info(f'############---------------Epoch {epoch} started.-----------------############')

        logging.info(f'Epoch {epoch} started.')
        start = time.time()
        try:
            # Train one epoch.
            fa_model, train_loss = train_epoch(epoch, train_dataloader, fa_model, criterion, optimizer, writer, use_cuda, lr)

            # Learning rate scheduled dropping.
            if train_loss > train_loss_best:
                lr_counter += 1
            if lr_counter > params.lr_patience:
                lr_counter = 0
                lr = lr/params.scheduled_drop
                logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                logging.info(f'Learning rate dropping to its {params.scheduled_drop}th value to {lr} at epoch {epoch}.')
                logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            # Set best train loss.
            if train_loss < train_loss_best:
                train_loss_best = train_loss

            # Validation epoch.
            if epoch in val_array:
                validation_dataset = reconstruction_dataset(data_split='test', shuffle=False, ucf101_percentage=1.0, data_percentage=1.0)
                validation_dataloader = DataLoader(validation_dataset, batch_size=48, shuffle=True, num_workers=params.num_workers)
                val_loss = validation_epoch(run_id, epoch, validation_dataloader, fa_model, criterion, writer, use_cuda)

            if val_loss < val_loss_best:
                logging.info('++++++++++++++++++++++++++++++')
                logging.info(f'Epoch {epoch} is the best model till now for {run_id}!')
                logging.info('++++++++++++++++++++++++++++++')

                save_file_path = os.path.join(save_dir, f'model_{epoch}_bestAcc_{val_loss:.6f}.pth')
                states = {
                    'epoch': epoch + 1,
                    'lr_counter' : lr_counter,
                    'fa_model_state_dict': fa_model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(states, save_file_path)
                val_loss_best = val_loss
            # else:
            save_dir = os.path.join(cfg.saved_models_dir, run_id)
            save_file_path = os.path.join(save_dir, 'model_temp.pth')
            states = {
                'epoch': epoch + 1,
                'lr_counter' : lr_counter,
                'fa_model_state_dict': fa_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
            # scheduler.step()
        except:
            logging.info(f'Epoch {epoch} failed.')
            logging.info('-'*60)
            # traceback.logging.info_exc(file=sys.stdout)
            logging.info('-'*60)
            continue
        time_taken = time.time() - start
        logging.info(f'Time taken for Epoch-{epoch} is {time_taken}')
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline.')

    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default='reconstruction_fa', help='run_id')
    parser.add_argument("--saved_model", dest='saved_model', type=str, required=False, default=None, help='run_id')
    args = parser.parse_args()
    logging.info(f'Run ID: {args.run_id}')

    train_classifier(args.run_id, args.saved_model)
