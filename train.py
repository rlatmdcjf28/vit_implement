import torch
import torch.nn as nn
from preprocessing import Patchfy
from dataset.load_dataset import CustomDataset
from models import Img2Patch, Projection_layer, ClassEmbedding, TransformerEncoder, FeedForwardNet
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/datasets/cifar10/')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--act_fn', type=str, default='GeLU')

    args = parser.parse_args()

    return args



def my_dataset():
    CustomDataset
    pass


def train():
    pass

def main(arg):
    pass


if __name__=='__main__':
    args = parse()
    
    breakpoint()

    main(args)
