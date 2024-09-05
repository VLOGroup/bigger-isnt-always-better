from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE, VESDE_CCDF
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_ct,
                      get_pc_fouriercs_ct_ccdf,
                      get_jalal_sampling_ct)
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
from physics.ct import *


def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
    N = args.N
    m = args.m

    if args.jalal.lower() in ('true', 't', 'yes', 'y', '1'):
        jalal = True
    else:
        jalal = False

    if args.head.lower() in ('true', 't', 'yes', 'y', '1'):
        head = True
    else:
        head = False

    if args.ccdf.lower() in ('true', 't', 'yes', 'y', '1'):
        ccdf = True
    else:
        ccdf = False

    print('initaializing...')
    configs = importlib.import_module(f'configs.ve.{args.model}')
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    np.random.seed(config.seed)
    
    ckpt_filename = f'/srv/local/lg/workdir/{args.model}/checkpoints/checkpoint.pth'
    if ccdf:
        sde = VESDE_CCDF(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N, N_prime=300)
    else:
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
    save_root = Path(f'/srv/local/lg/{"jalal_new" if jalal else "results_new"}/{args.model}' + f'_{args.problem}' + f'_a_{args.angle}' + f'_n_{args.num_views}' + ('' if args.N == 2000 else f'_N_{args.N}') + ('_CCDF' if ccdf else '') + ('_head' if head else '') + '_norm')
    save_root.mkdir(parents=True, exist_ok=True)

    if args.problem == 'lim_angle':
        ct = CT(img_width=320, n_views=args.num_views, end_angle=args.angle, circle=False, device=config.device)
    elif args.problem == 'few_view':
        ct = CT(img_width=320, n_views=args.num_views, circle=False, device=config.device)
    else:
        raise TypeError('Wrong Problem')


    ssim = SSIM().cuda()

    if head:
        test_dl = mydata.create_ct_head_dataloader(320, batch_size=1)
    else:
        _, test_dl = mydata.create_ct_dataloader(320, batch_size=1)

    psnr_values = np.zeros(len(test_dl))
    ssim_values = np.zeros(len(test_dl))
    nmse_values = np.zeros(len(test_dl))
    prediction_times = np.zeros((len(test_dl)))

    ###############################################
    # 2. Inference
    ###############################################
    if ccdf:
        pc_fouriercs = get_pc_fouriercs_ct_ccdf(sde,
                                            predictor, corrector,
                                            inverse_scaler,
                                            snr=snr,
                                            ct=ct,
                                            n_steps=m,
                                            probability_flow=probability_flow,
                                            continuous=config.training.continuous,
                                            denoise=True)
    elif jalal:
        pc_fouriercs = get_jalal_sampling_ct(sde,
                                            predictor, corrector,
                                            inverse_scaler,
                                            snr=snr,
                                            ct=ct,
                                            n_steps=m,
                                            probability_flow=probability_flow,
                                            continuous=config.training.continuous,
                                            denoise=True)
    else:
        pc_fouriercs = get_pc_fouriercs_ct(sde,
                                            predictor, corrector,
                                            inverse_scaler,
                                            snr=snr,
                                            ct=ct,
                                            n_steps=m,
                                            probability_flow=probability_flow,
                                            continuous=config.training.continuous,
                                            denoise=True)
    
    torch.manual_seed(42)

    for i, img in enumerate(test_dl):
        print(f'reconstructing slice {i + 1} of {len(test_dl)}')
        img = img.view(1, 1, 320, 320)
        img = img.to(config.device)

        img -= torch.min(img)
        img /= torch.max(img)

        sino = ct.A(img)
        under_img = ct.AT(sino)
        print(sino.shape)
        plt.imsave(str(save_root) + f'/img_{i}.png', img.squeeze().detach().cpu().numpy(), cmap='gray')
        np.save(str(save_root) + f'/img_{i}', img.squeeze().detach().cpu().numpy())
        plt.imsave(str(save_root) + f'/sino_{i}.png', sino.squeeze().detach().cpu().numpy(), cmap='gray')
        plt.imsave(str(save_root) + f'/under_img_{i}.png', under_img.squeeze().detach().cpu().numpy(), cmap='gray')
        np.save(str(save_root) + f'/under_img_{i}', under_img.squeeze().detach().cpu().numpy())
        plt.imsave(str(save_root) + f'/dag_img_{i}.png', ct.A_dagger(sino).squeeze().detach().cpu().numpy(), cmap='gray')


        # undersampling

        tic = time.time()
        x = pc_fouriercs(score_model, scaler(under_img), Fy=sino)
        prediction_times[i] = time.time() - tic
        psnr_values[i] = psnr(x, img)
        nmse_values[i] = nmse(x, img)
        ssim_values[i] = ssim(x, img)
        np.savetxt(os.path.join(save_root, 'prediction_times.csv'), prediction_times, delimiter=',')
        np.savetxt(os.path.join(save_root, 'psnr_values.csv'), psnr_values, delimiter=',')
        np.savetxt(os.path.join(save_root, 'nmse_values.csv'), nmse_values, delimiter=',')
        np.savetxt(os.path.join(save_root, 'ssim_values.csv'), ssim_values, delimiter=',')
        recon = x.squeeze().cpu().detach().numpy()
        plt.imsave(str(save_root) + f'/recon_{i}.png', recon, cmap='gray', vmin=0, vmax=1)
        np.save(str(save_root) + f'/recon_{i}', recon)
        if i == 99:
            break
    
    print(f'Number of Parameters: {sum(p.numel() for p in score_model.parameters())}')
    print(f'Number of Trainable Parameters: {sum(p.numel() for p in score_model.parameters() if p.requires_grad)}')


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ccdf', type=str, help='Whether reconstructing complex- or real-valued data.', default='False')
    parser.add_argument('--head', type=str, help='Whether reconstructing complex- or real-valued data.', default='False')
    parser.add_argument('--jalal', type=str, help='Whether reconstructing complex- or real-valued data.', default='False')
    parser.add_argument('--model', type=str, help='which config file to use', required=True)
    parser.add_argument('--problem', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='lim_angle',
                        choices=['few_view', 'lim_angle'])
    parser.add_argument('--angle', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=1)
    parser.add_argument('--num_views', type=int, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=180)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--N', type=int, help='Number of iterations for score-POCS sampling', default=2000)
    parser.add_argument('--m', type=int, help='Number of corrector step per single predictor step.'
                                              'It is advised not to change this default value.', default=1)
    return parser


if __name__ == "__main__":
    main()