import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.optim import lr_scheduler
import torchsummary
from torchsummary import summary
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

import argparse
import io
import sys
import os
import time
import random

## Load Custom Modules
from preprocessing import Patchfy
from dataset.load_dataset import CustomDataset, custom_dataloader
from models import ViTModel
from performance_measure import loss_plot, accuracy_plot, confusionmatrix, \
                                report, roc_n_auc, learning_rate_plot

from torch_lr_scheduler import CosineAnnealingWarmUpRestarts as CAWR


## random seed fix
def random_seed_fix(random_seed):
    random.seed(random_seed) # torchvision randomcrop, ...
    torch.manual_seed(random_seed)


def parse():
    parser = argparse.ArgumentParser()
    
    ## general parameters
    parser.add_argument('--training_device',   type=str,   default='lab')
    parser.add_argument('--dataset_dir',       type=str,   default='/Data_RTX4090_server/datasets/cifar10/')
    parser.add_argument('--epochs',            type=int,   default=1)
    parser.add_argument('--batch_size',        type=int,   default=1)
    parser.add_argument('--learning_rate',     type=float, default=1e-8)
    parser.add_argument('--optimizer',         type=str,   default='Adam')
    parser.add_argument('--act_fn',            type=str,   default='GeLU')
    parser.add_argument('--device',            type=str,   default='cuda')
    parser.add_argument('--dropout_rate',      type=float, default=0.1)
    parser.add_argument('--random_seed',       type=int,   default=123)
    parser.add_argument('--model_summary_dir', type=str,   default='model_summary.txt')
    parser.add_argument('--result_dir',        type=str,   default='./result_dir/'+time.strftime("%Y%m%d_%H%M")+'/')
    
    parser.add_argument('--classification_report_dir', type=str, default='classification_report.txt')
    
    parser.add_argument('--now_epochs', type=int, default=1)

    ## model parameters
    parser.add_argument('--patch_size',     type=int, default=8)
    parser.add_argument('--num_enc_layers', type=int, default=2)
    parser.add_argument('--mlp_size',       type=int, default=512)
    parser.add_argument('--hidden_dim',     type=int, default=256)
    parser.add_argument('--num_heads',      type=int, default=4)
    parser.add_argument('--num_classes',    type=int, default=10)
 
    args = parser.parse_args()

    return args



def custom_dataset(args):

    train_dataset = CustomDataset(root_dir=args.dataset_dir+'train')
    test_dataset  = CustomDataset(root_dir=args.dataset_dir+'test')
    
    class_list = train_dataset.class_names_list()

    train_size = int(len(train_dataset) * 0.8)
    val_size   = int(len(train_dataset) * 0.2)
    
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


    train_dataloader, val_dataloader, test_dataloader = custom_dataloader(args.batch_size,
                                                                           train_dataset,
                                                                           val_dataset,
                                                                           test_dataset)
    
    return train_dataloader, val_dataloader, test_dataloader, class_list



## with tqdm
def train_loop(args, device, train_dataloader, model, criterion, optim, scheduler, 
               _train_loss_list=[],_train_acc_list=[]):
    _train_loss_list = []
    _train_acc_list = []
    total_train_loss = 0.0
    train_correct = 0
    train_total = 0
    train_lr_list = []

    train_progress_bar = tqdm(train_dataloader, desc=f'Epoch {args.now_epochs}/{args.epochs}')

    model.train()
    for imgs, labels in train_progress_bar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        outputs = model(imgs)
        
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        
        optim.step()
        optim.zero_grad()
        
        total_train_loss += train_loss.item()
        _, pred = torch.max(outputs.data, 1)
        
        train_total += labels.size(0)
        train_correct += (pred == labels).sum().item()

        train_loss = total_train_loss / len(train_dataloader)
        train_acc  = 100 * train_correct / train_total
        
    train_progress_bar.set_postfix(loss=f'{train_loss:.4f}', acc=f'{train_acc:.2f}')

    
    scheduler.step()
        
    print('\n')
    print(f'           Train_loss      : {train_loss:.5f},  Train_accuracy      : {train_acc:.3f}')
    

    return train_loss, train_acc

def validation_loop(args, device, val_dataloader, model, criterion, optim):
    total_val_loss = 0.0
    val_correct = 0
    val_total = 0

    model.eval()
    with torch.no_grad():
        for imgs, labels in val_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
            total_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        val_loss = total_val_loss / len(val_dataloader)
        val_acc  = 100 * val_correct / val_total
        

    print(f'           Validation loss : {val_loss:.5f},  Validation accuracy : {val_acc:.3f}')
    print('\n')
    
    return val_loss, val_acc


def test_loop(args, model, device, dataloader, criterion):
    test_loss = 0.0
    correct   = 0
    total     = 0
    
    gt_labels   = []
    pred_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            model.eval()
            outputs = model(imgs)
            
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            gt_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    loss = test_loss / len(dataloader)
    accuracy = 100 * correct / total
    print('Testing Time!')
    print('\n')
    print(f'           Test Acc : {accuracy}')
    print('\n')
    
    return pred_labels, gt_labels


def loss_fn():
    criterion = nn.CrossEntropyLoss()

    return criterion


def optimizer(args, model): # default = Adam
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.03)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    
    ## Need to think about how to make it more flexible.

    return optimizer


def learning_scheduler(optim):
    breakpoint()
    linear_lr = lr_scheduler.LinearLR(optim, total_iters=10)
    step_lr   = lr_scheduler.StepLR(optim, step_size=30, gamma=0.9)
    scheduler = lr_scheduler.SequentialLR(optim, 
                                          milestones=[10],
                                          schedulers = \
                                          [linear_lr, step_lr])
    
    # if using CosineAnnealingWarmUpRestarts, initial learning rate should be very small.
    # cawr = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0)
    # scheduler = CAWR(optim,)
    return scheduler


def model_to_txt(model, args):
    if os.path.exists(args.result_dir+args.model_summary_dir):
        os.remove(args.result_dir+args.model_summary_dir)

    summary_str = io.StringIO()
    sys.stdout = summary_str
    summary(model, input_data=(3, 32, 32), col_names=('input_size', 'output_size', 'num_params'))
    sys.stdout = sys.__stdout__
    with open(args.result_dir+args.model_summary_dir, "w") as f:
        f.write(summary_str.getvalue())


def main(args): 
    train_dataloader, val_dataloader, test_dataloader, class_list = custom_dataset(args)

    model = ViTModel(args).to(args.device)

    model_to_txt(model, args)

    _train_loss_list = []
    _train_acc_list  = []
    _val_loss_list   = []
    _val_acc_list    = []
    lr_list          = []
    
    scheduler = learning_scheduler(optim=optimizer(args, model))

    for _ in range(args.epochs):
        # breakpoint()
        train_loss, train_acc = train_loop(args,
                                           args.device,
                                           train_dataloader, 
                                           model, 
                                           criterion=loss_fn(),
                                           optim=optimizer(args, model),
                                           scheduler=scheduler)
        
        val_loss, val_acc = validation_loop(args,
                                            args.device, 
                                            val_dataloader,
                                            model,
                                            criterion=loss_fn(),
                                            optim=optimizer(args, model))
        print(scheduler.get_last_lr())
        lr_list.extend(scheduler.get_last_lr())
        args.now_epochs += 1
        _train_loss_list.append(train_loss)
        _train_acc_list.append(train_acc)
        _val_loss_list.append(val_loss)
        _val_acc_list.append(val_acc)

    pred_labels, gt_labels = test_loop(args, 
                                       model, 
                                       args.device, 
                                       test_dataloader, 
                                       #criterion=loss_fn()
                                       criterion=nn.CrossEntropyLoss())

    train_loss_list = [round(loss, 5) for loss in _train_loss_list]
    train_acc_list  = [round(acc, 3) for acc in _train_acc_list]
    val_loss_list   = [round(loss, 5) for loss in _val_loss_list]
    val_acc_list    = [round(acc, 3) for acc in _val_acc_list]

    # breakpoint()
    
    learning_rate_plot(args, lr_list=lr_list)

    loss_plot(train_loss_list, val_loss_list, args)

    accuracy_plot(train_acc_list, val_acc_list, args)

    confusionmatrix(gt_labels, pred_labels, args)

    report(args, gt_labels, pred_labels, class_list)

    roc_n_auc(args, gt_labels, pred_labels, class_list)
    
    breakpoint()

    a = 0

def param_save(args):
    with open(args.result_dir+'parameters.txt', "w") as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')

if __name__=='__main__':
    args = parse()
    
    if os.path.exists(args.result_dir):
        os.remove(os.path(args.result_dir))

    os.makedirs(args.result_dir)

    random_seed_fix(args.random_seed)

    if args.device == 'cuda':
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    
    if args.training_device == 'lab':
        args.dataset_dir = '/Data_RTX4090_server/datasets/cifar10/'
    else:
        args.dataset_dir = '/datasets/cifar10/'
    
    param_save(args)

    # breakpoint()

    main(args)

    breakpoint()

    a = 0