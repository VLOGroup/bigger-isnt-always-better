from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_RI,
                      get_pc_fouriercs_fast)
from models import ncsnpp
import time
from utils import (fft2, ifft2, get_mask, get_data_scaler, 
                   get_data_inverse_scaler, restore_checkpoint, 
                   SSIM, nmse, psnr, get_radial_mask)
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
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
    img_size = config.data.image_size
    batch_size = 1

    np.random.seed(config.seed)
    if args.mask_type == 'radial':
        mask = get_radial_mask((320, 320), 29, np.pi / 29)
    else:
        mask = get_mask(torch.zeros((1, 1, 320, 320)), img_size, batch_size,
                        type=args.mask_type,
                        acc_factor=args.acc_factor,
                        center_fraction=args.center_fraction).to(config.device)
    
    ckpt_filename = f'/srv/local/lg/workdir/{args.model}/checkpoints/checkpoint.pth'
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N)

    config.training.batch_size = batch_size
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    probability_flow = False
    snr = 0.16

    # sigmas = mutils.get_sigmas(config)
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # create model and load checkpoint
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)
    state = restore_checkpoint(ckpt_filename, state, config.device) #  , skip_sigma=True)
    ema.copy_to(score_model.parameters())

    print(f'Number of Parameters: {sum(p.numel() for p in score_model.parameters())}')
    print(f'Number of Trainable Parameters: {sum(p.numel() for p in score_model.parameters() if p.requires_grad)}')

    # Specify save directory for saving generated samples
    print(args.center_fraction)
    save_root = Path(f'/srv/local/lg/results/{args.model}' + ('' if args.mask_type == 'gaussian1d' else f'_{args.mask_type}') + ('' if args.center_fraction == 0.08 else f'_CF_0{str(args.center_fraction)[2:]}') + ('' if args.acc_factor == 4 else f'_AF_{args.acc_factor}') + ('' if args.N == 2000 else f'_N_{args.N}') + ('_FS' if fat_suppression else '') + ('_brain' if brain else '') + '_no_norm')
    save_root.mkdir(parents=True, exist_ok=True)

    mask_sv = mask.squeeze().cpu().detach().numpy()

    np.save(str(save_root) + '/mask.npy', mask_sv)
    plt.imsave(str(save_root) + '/mask.png', mask_sv, cmap='gray')

    ssim = SSIM().cuda()
    if brain:
        test_dl = mydata.create_brain_dataloader()
    else:
        _, test_dl = mydata.create_dataloader(config, fat_suppression=fat_suppression)
    psnr_values = np.zeros(len(test_dl))
    ssim_values = np.zeros(len(test_dl))
    nmse_values = np.zeros(len(test_dl))
    prediction_times = np.zeros((len(test_dl)))

    ###############################################
    # 2. Inference
    ###############################################

    if use_complex:
        pc_fouriercs = get_pc_fouriercs_RI(sde,
                                        predictor, corrector,
                                        inverse_scaler,
                                        snr=snr,
                                        n_steps=m,
                                        probability_flow=probability_flow,
                                        continuous=config.training.continuous,
                                        denoise=True)
    else:
        pc_fouriercs = get_pc_fouriercs_fast(sde,
                                         predictor, corrector,
                                         inverse_scaler,
                                         snr=snr,
                                         n_steps=m,
                                         probability_flow=probability_flow,
                                         continuous=config.training.continuous,
                                         denoise=True)
    
    for i, img in enumerate(test_dl):
        print(f'reconstructing slice {i + 1} of {len(test_dl)}')
        img = img.view(1, 1, 320, 320)
        img = img.to(config.device)
        img = vertical_flip(img)
        kspace = fft2(img)

        # undersampling
        under_kspace = kspace * mask
        under_img = ifft2(under_kspace)
        if not use_complex:
            img = torch.real(img)
            under_img = torch.real(under_img)

        tic = time.time()
        x = pc_fouriercs(score_model, scaler(under_img), mask, Fy=under_kspace)
        prediction_times[i] = time.time() - tic
        psnr_values[i] = psnr(x, img)
        nmse_values[i] = nmse(x, img)
        ssim_values[i] = ssim(x, img)
        np.savetxt(os.path.join(save_root, 'prediction_times.csv'), prediction_times, delimiter=',')
        np.savetxt(os.path.join(save_root, 'psnr_values.csv'), psnr_values, delimiter=',')
        np.savetxt(os.path.join(save_root, 'nmse_values.csv'), nmse_values, delimiter=',')
        np.savetxt(os.path.join(save_root, 'ssim_values.csv'), ssim_values, delimiter=',')
        recon = x.squeeze().cpu().detach().numpy()
        plt.imsave(str(save_root) + f'/recon_{i}.png', recon, cmap='gray')
        np.save(str(save_root) + f'/recon_{i}', recon)
    
    print(f'Number of Parameters: {sum(p.numel() for p in score_model.parameters())}')
    print(f'Number of Trainable Parameters: {sum(p.numel() for p in score_model.parameters() if p.requires_grad)}')


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--complex', type=str, help='Whether reconstructing complex- or real-valued data.', default='False')
    parser.add_argument('--fat_suppression', type=str, help='Whether reconstructing CORPDFS or CORPD data.', default='False')
    parser.add_argument('--brain', type=str, help='Whether reconstructing brain data.', default='False')
    parser.add_argument('--model', type=str, help='which config file to use', required=True)
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='gaussian1d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d', 'poisson', 'radial'])
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