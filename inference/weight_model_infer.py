import argparse
import os
import sys
sys.path.append('../')

import imageio
import numpy as np
import torch
from scipy import ndimage
from torchvision.utils import save_image

from models.auto_encoder import AE
from models.variational_autoencoder import VAE
from utils import get_interpolations

parser = argparse.ArgumentParser(
  description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--models', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                    help='Which dataset to use')


if __name__ == "__main__":

  print("[trace] train.py@infer")
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  torch.manual_seed(args.seed)
  vae = VAE(args)
  ae = AE(args)

  try:
    architectures = {'AE': ae, 'VAE': vae}
    print(f"[trace] current selected model: {args.models}")
    print(f"[trace] retrieve the target model")
    autoenc = architectures[args.models]
  except KeyError:
    print('---------------------------------------------------------')
    print('Model architecture not supported. ', end='')
    print('Maybe you can implement it?')
    print('---------------------------------------------------------')
    sys.exit()

  print('[trace] run@main start to save the models')
  path = './trained'
  autoenc.restore_from_weight_file(path)
  input_path = './results/one.png'
  if (os.path.exists(input_path) == False):
    print(f'[trace] target file {input_path} does not exist, exit')
    exit(-1)

  print('[trace] start to infer the embeddings for the input')
  device = torch.device("cuda" if args.cuda else "cpu")
  print(f'[trace] current device is {device}')
  result = autoenc.infer(input_path, device)
  print(f'[trace] the shape of this embedding tensor is {result.size()}')
  print(f'[trace] the value of this embedding tensor is {result}')
