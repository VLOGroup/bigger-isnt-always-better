from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_RI,
                      get_pc_fouriercs_fast)
from models import ncsnpp
import time
from utils import fft2, ifft2, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint, normalize_complex, normalize, ifft2_m, fft2_m, SSIM, nmse, psnr, get_radial_mask, get_outer_mask
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
import h5py
from torchvision.transforms.functional import center_crop
from torchvision.transforms.v2.functional import vertical_flip
import mydata
import os


def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
    N = args.N
    m = args.m
    if args.complex.lower() in ('true', 't', 'yes', 'y', '1'):
        use_complex = True
    else:
        use_complex = False
    if args.fat_suppression.lower() in ('true', 't', 'yes', 'y', '1'):
        fat_suppression = True
    else:
        fat_suppression = False
    if args.brain.lower() in ('true', 't', 'yes', 'y', '1'):
        brain = True
        assert fat_suppression == False
    else:
        brain = False

    print('initaializing...')
    configs = importlib.import_module(f'configs.ve.{args.model}')
    config = configs.get_config()


    save_root = Path(f'./labels/' + ('CORPD' if not (fat_suppression or brain) else '') + ('CORPDFS' if fat_suppression else '') + ('brain' if brain else ''))
    save_root.mkdir(parents=True, exist_ok=True)

    if brain:
        test_dl = mydata.create_brain_dataloader()
    else:
        _, test_dl = mydata.create_dataloader(config, fat_suppression=fat_suppression)
    ###############################################
    # 2. Inference
    ###############################################
    
    for i, img in enumerate(test_dl):
        print(f'reconstructing slice {i + 1} of {len(test_dl)}')
        # fft
        # plt.imsave(str(save_root) + f'/input_{i}.png', img.squeeze().cpu().detach().numpy(), cmap='gray')
        img = img.view(1, 1, 320, 320)
        img = vertical_flip(img)

        plt.imsave(str(save_root) + f'/label_{i}.png', img.squeeze().cpu().detach().numpy(), cmap='gray')
        np.save(str(save_root) + f'/label_{i}', img.squeeze().cpu().detach().numpy())

    ###############################################
    # 3. Saving recon
    ###############################################
    


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--complex', type=str, help='Whether reconstructing complex- or real-valued data.', default='False')
    parser.add_argument('--fat_suppression', type=str, help='Whether reconstructing CORPDFS or CORPD data.', default='False')
    parser.add_argument('--brain', type=str, help='Whether reconstructing brain data.', default='False')
    parser.add_argument('--model', type=str, help='which config file to use', required=True)
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='gaussian1d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d', 'poisson', 'radial', 'outer'])
    parser.add_argument('--cutout', type=int, help='Size of the cutout if outer mask is used.', default=0)
    parser.add_argument('--acc_factor', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=4)
    parser.add_argument('--center_fraction', type=float, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=0.08)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--N', type=int, help='Number of iterations for score-POCS sampling', default=2000)
    parser.add_argument('--m', type=int, help='Number of corrector step per single predictor step.'
                                              'It is advised not to change this default value.', default=1)
    return parser


if __name__ == "__main__":
    main()