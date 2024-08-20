import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

import torchsummary
from torchsummary import summary
from tqdm import tqdm
from preprocessing import Patchfy
from dataset.load_dataset import CustomDataset, custom_dataloader
from models import ViTModel
import argparse

import io
import sys
import os

import random

## random seed fix
def random_seed_fix(random_seed):
    random.seed(random_seed) # torchvision randomcrop, ...
    torch.manual_seed(random_seed)


def parse():
    parser = argparse.ArgumentParser()
    
    ## general parameters
    parser.add_argument('--training_device',   type=str,   default='lab')
    parser.add_argument('--dataset_dir',       type=str,   default='/home/ai/User/seungchul/cifar10/')
    parser.add_argument('--epochs',            type=int,   default=1)
    parser.add_argument('--batch_size',        type=int,   default=1)
    parser.add_argument('--learning_rate',     type=float, default=0.001)
    parser.add_argument('--optimizer',         type=str,   default='Adam')
    parser.add_argument('--act_fn',            type=str,   default='GeLU')
    parser.add_argument('--device',            type=str,   default='cpu')
    parser.add_argument('--dropout_rate',      type=float, default=0.3)
    parser.add_argument('--random_seed',       type=int,   default=123)
    parser.add_argument('--model_summary_dir', type=str,   default='./model_summary.txt')
    
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
    
    # breakpoint()

    train_dataset = CustomDataset(root_dir=args.dataset_dir+'train')
    test_dataset  = CustomDataset(root_dir=args.dataset_dir+'test')
    
    train_size = int(len(train_dataset) * 0.8)
    val_size   = int(len(train_dataset) * 0.2)
    
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


    train_dataloader, val_dataloader, test_dataloader = custom_dataloader(args.batch_size,
                                                                           train_dataset,
                                                                           val_dataset,
                                                                           test_dataset)
    
    return train_dataloader, val_dataloader, test_dataloader



    
'''
def train_n_val_loop(args, train_dataloader, val_dataloader, model, criterion, optim):
    for epoch in range(args.epochs):
        total_train_loss = 0.0
        total_val_loss   = 0.0

        train_correct = 0
        val_correct   = 0
        
        train_total = 0
        val_total   = 0
        
        ## Training loop
        for imgs, labels in train_dataloader:
            model.train()
            optim.zero_grad()
            outputs = model(imgs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optim.step()
            total_train_loss += train_loss.item()
            _, pred = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()
        
        train_loss = train_loss/len(train_dataloader)
        train_acc  = 100 * train_correct / train_total

        ## Validation loop
        with torch.no_grad():
            for imgs, labels in val_dataloader:
                model.eval()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

            val_loss = total_val_loss / len(val_dataloader)
            val_acc = 100 * val_correct / val_total

        print(f'Epoch : {epoch+1}, Train_loss : {train_loss}, Train_accuracy : {train_acc}')
        print(f'                   Validation loss : {val_loss}, Validation accuracy : {val_acc}')


def test_loop(model, dataloader, criterion):
    test_loss = 0.0
    correct   = 0
    total     = 0

    with torch.no_grad():
        for img, labels in dataloader:
            model.eval()
            outputs = model(img)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = test_loss / len(dataloader)
    accuracy = 100 * correct / total
    return loss, accuracy
'''


## with tqdm
def train_n_val_loop(args, device, train_dataloader, val_dataloader, model, criterion, optim):
    _train_loss_list = []
    _val_loss_list   = []

    _train_acc_list = []
    _val_acc_list   = []

    for epoch in range(args.epochs):
        total_train_loss = 0.0
        total_val_loss   = 0.0

        train_correct = 0
        val_correct   = 0
        
        train_total = 0
        val_total   = 0
        
        train_progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        ## Training loop
        for imgs, labels in train_progress_bar:
            
            imgs, labels = imgs.to(device), labels.to(device)

            model.train()
            
            optim.zero_grad()
            
            outputs = model(imgs)
            
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optim.step()

            total_train_loss += train_loss.item()
            _, pred = torch.max(outputs.data, 1)
            
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()
        
        train_loss = total_train_loss / len(train_dataloader)
        train_acc  = 100 * train_correct / train_total
        
        train_progress_bar.set_postfix(loss=f'{train_loss:.4f}', acc=f'{train_acc:.2f}')

        _train_loss_list.append(train_loss)
        _train_acc_list.append(train_acc)

        train_loss_list = [round(loss, 5) for loss in _train_loss_list]
        train_acc_list = [round(acc, 3) for acc in _train_acc_list]

        ## Validation loop
        with torch.no_grad():
            for imgs, labels in val_dataloader:

                imgs, labels = imgs.to(device), labels.to(device)
                
                model.eval()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

            val_loss = total_val_loss / len(val_dataloader)
            val_acc  = 100 * val_correct / val_total
            
            _val_loss_list.append(val_loss)
            _val_acc_list.append(val_acc)

            val_loss_list = [round(loss, 5) for loss in _val_loss_list]
            val_acc_list = [round(acc, 5) for acc in _val_acc_list]

        print(f'        Train_loss      : {train_loss:.4f}, Train_accuracy      : {train_acc:.2f}')
        print(f'        Validation loss : {val_loss:.4f}, Validation accuracy : {val_acc:.2f}')
        print('\n')
    
    return train_loss_list, train_acc_list, val_loss_list, val_acc_list


def test_loop(model, dataloader, criterion):
    test_loss = 0.0
    correct   = 0
    total     = 0

    with torch.no_grad():
        for img, labels in dataloader:
            model.eval()
            outputs = model(img)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = test_loss / len(dataloader)
    accuracy = 100 * correct / total
    return loss, accuracy


def loss_fn():
    criterion = nn.CrossEntropyLoss()

    return criterion


def optim(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return optimizer

def model_to_txt(model, args):
    if os.path.exists(args.model_summary_dir):
        os.remove(args.model_summary_dir)

    summary_str = io.StringIO()
    sys.stdout = summary_str
    summary(model, input_data=(3, 32, 32), col_names=('input_size', 'output_size', 'num_params'))
    sys.stdout = sys.__stdout__
    with open(args.model_summary_dir, "w") as f:
        f.write(summary_str.getvalue())

def main(args): 
    train_dataloader, val_dataloader, test_dataloader = custom_dataset(args)

    model = ViTModel(args).to(args.device)

    model_to_txt(model, args)
    
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = train_n_val_loop(args,
                                                                                    args.device,
                                                                                    train_dataloader, 
                                                                                    val_dataloader,
                                                                                    model, 
                                                                                    criterion=loss_fn(), 
                                                                                    optim=optim(args, model))

    
    breakpoint()

    test_loop(model, test_dataloader, criterion=loss_fn())



if __name__=='__main__':
    args = parse()
    
    random_seed_fix(args.random_seed)

    if args.device == 'cuda':
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    
    if args.training_device == 'lab':
        args.dataset_dir = '/home/ai/User/seungchul/cifar10/'
    else:
        args.dataset_dir = '/datasets/cifar10/'
    
    # breakpoint()

    main(args)

    breakpoint()

    a = 0