from pathlib import Path
import time
from utils import (fft2, ifft2, get_mask,
                   SSIM, nmse, psnr, get_radial_mask,
                   AsMatrix, inner, Div, Grad, apgd, CharbTV,
                   unet_model, unet_normalize, call_unet)
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
from torchvision.transforms.functional import center_crop
import mydata
import os
import fastmri.models as fm
from typing import Union, Tuple, Callable
from physics.ct import *

def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
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
    if args.ct.lower() in ('true', 't', 'yes', 'y', '1'):
        ct = True
    else:
        ct = False
    if args.ct_head.lower() in ('true', 't', 'yes', 'y', '1'):
        ct_head = True
    else:
        ct_head = False
    if args.head.lower() in ('true', 't', 'yes', 'y', '1'):
        head = True
    else:
        head = False

    print('initaializing...')
    configs = importlib.import_module(f'configs.ve.fastmri_knee_4_attention')
    config = configs.get_config()
    img_size = config.data.image_size
    config.data.batch_size = 1
    batch_size = 1

    np.random.seed(config.seed)
    if args.mask_type == 'radial':
        mask = get_radial_mask((320, 320), 29, np.pi / 29)
    else:
        mask = get_mask(torch.zeros((batch_size, 1, 320, 320)), img_size, batch_size,
                        type=args.mask_type,
                        acc_factor=args.acc_factor,
                        center_fraction=args.center_fraction).to(config.device)

    # Specify save directory for saving generated samples
    print(args.center_fraction)
    if ct:
        save_root = Path(f'/srv/local/lg/results_new_no_scale/{args.model}' + f'_{args.problem}' + f'_a_{args.angle}' + f'_n_{args.num_views}' + ('' if args.N == 2000 else f'_N_{args.N}') + ('_head' if head else '') + '_no_norm')
    else:
        save_root = Path(f'/srv/local/lg/results_new_no_scale/{args.model}' + ('' if args.mask_type == 'gaussian1d' else f'_{args.mask_type}') + ('' if args.center_fraction == 0.08 else f'_CF_0{str(args.center_fraction)[2:]}') + ('' if args.acc_factor == 4 else f'_AF_{args.acc_factor}') + ('' if args.N == 2000 else f'_N_{args.N}') + ('_FS' if fat_suppression else '') + ('_brain' if brain else '') + '_no_norm')
    save_root.mkdir(parents=True, exist_ok=True)
    if ct:
        lambdas = [2.5, 5, 7.5, 10] ##  [0.75, 1, 2.5]  # [5, 7.5, 10, 15, 20]
    else:
        lambdas = [10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 5000, 7500, 10000, 25000, 50000, 100000, 200000]

    ssim = SSIM().cuda()
    if ct:
        if head:
            test_dl = mydata.create_ct_head_dataloader(320, batch_size=1)
        else:
            _, test_dl = mydata.create_ct_dataloader(320, batch_size=1)
    else:
        if brain:
            test_dl = mydata.create_brain_dataloader()
        else:
            _, test_dl = mydata.create_dataloader(config, fat_suppression=fat_suppression)
    
    if args.model == 'tv':
        psnr_values = np.zeros((len(test_dl) * batch_size, len(lambdas)))
        ssim_values = np.zeros((len(test_dl) * batch_size, len(lambdas)))
        nmse_values = np.zeros((len(test_dl) * batch_size, len(lambdas)))
    else:
        psnr_values = np.zeros((len(test_dl) * batch_size))
        ssim_values = np.zeros((len(test_dl) * batch_size))
        nmse_values = np.zeros((len(test_dl) * batch_size))

    unet = unet_model('/srv/local/lg/models/unet.ckpt')
    print(f'Number of Parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}')
    
    pred_times = np.zeros((100, 1))

    R = CharbTV(eps=1e-3)

    if ct:
        if args.problem == 'lim_angle':
            ct = CT(img_width=320, n_views=args.num_views, end_angle=args.angle, circle=False, device=config.device)
        elif args.problem == 'few_view':
            ct = CT(img_width=320, n_views=args.num_views, circle=False, device=config.device)


    ###############################################
    # 2. Inference
    ###############################################

    for i, img in enumerate(test_dl):
        if i != 12:
            continue
        print(f'reconstructing slice {i + 1} of {len(test_dl)}')
        # fft
        # plt.imsave(str(save_root) + f'/input_{i}.png', img.squeeze().cpu().detach().numpy(), cmap='gray')
        img = img.view(batch_size, 1, 320, 320)
        img = img.to(config.device)
        # img -= torch.min(img)
        # img /= torch.max(img)
        kspace = fft2(img)

        # undersampling
        under_kspace = kspace * mask
        under_img = ifft2(under_kspace)
        if not use_complex:
            img = torch.real(img)
            under_img = torch.real(under_img)

        if args.model == 'unet':
            x = call_unet(unet, under_img)
            psnr_values[i*batch_size:(i+1)*batch_size] = psnr(x, img).squeeze().cpu().detach().numpy()
            nmse_values[i*batch_size:(i+1)*batch_size] = nmse(x, img).squeeze().cpu().detach().numpy()
            ssim_values[i*batch_size:(i+1)*batch_size] = ssim(x, img).squeeze().cpu().detach().numpy()

        else:
            tic = time.time()
            for j, lamda in enumerate(lambdas):
                if ct or ct_head:
                    A = AsMatrix(
                        lambda x: ct.A(x),
                        lambda y: ct.AT(y),
                    )
                else:
                    A = AsMatrix(
                        lambda x: mask * fft2(x),
                        lambda y: ifft2(mask * y).real,
                    )
                p = A @ img
                print(torch.sum(p==0))

                f_init = img.view(*img.shape[:2], img.shape[2] * img.shape[3])[..., torch.randperm(img.shape[2] * img.shape[3])].view(img.shape).clone()
                print(f_init.shape)
                def f_nabla(f):
                    dterm = lamda * ((A @ f - p).abs()**2).sum((1, 2, 3)) / 2
                    nabla_dterm = lamda * (A.H @ (A @ f - p))
                    reg, nabla_reg = R.grad(f)
                    return (
                        dterm[:, None, None, None] + reg,
                        nabla_dterm + nabla_reg,
                    )
                tv = apgd(
                    f_init,
                    f_nabla,
                    lambda x: f_nabla(x)[0],
                    lambda x, _: x,
                    max_iter=200,
                    gamma=0,
                )
                # This only improves TV
                # x = tv / tv.amax((1, 2, 3), keepdim=True)
                x = tv
                if args.model == 'tv':
                    psnr_values[i*batch_size:(i+1)*batch_size, j] = psnr(x, img).squeeze().cpu().detach().numpy()
                    nmse_values[i*batch_size:(i+1)*batch_size, j] = nmse(x, img).squeeze().cpu().detach().numpy()
                    ssim_values[i*batch_size:(i+1)*batch_size, j] = ssim(x, img).squeeze().cpu().detach().numpy()
                pred_times[i] = time.time()  - tic
                recon = x.squeeze().cpu().detach().numpy()
                # plt.imsave(str(save_root) + f'/recon_l_{lamda}_{i}.png', recon, cmap='gray')
                np.save(str(save_root) + f'/recon_l_{lamda}_{i}', recon)
        np.savetxt(os.path.join(save_root, 'psnr_values.csv'), psnr_values, delimiter=',')
        np.savetxt(os.path.join(save_root, 'nmse_values.csv'), nmse_values, delimiter=',')
        np.savetxt(os.path.join(save_root, 'ssim_values.csv'), ssim_values, delimiter=',')
        np.savetxt(os.path.join(save_root, 'prediction_times.csv'), pred_times, delimiter=',')
        if i == 99:
            print(np.mean(psnr_values, axis=0)[:10, :])
            break

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--complex', type=str, help='Whether reconstructing complex- or real-valued data.', default='False')
    parser.add_argument('--fat_suppression', type=str, help='Whether reconstructing CORPDFS or CORPD data.', default='False')
    parser.add_argument('--brain', type=str, help='Whether reconstructing brain data.', default='False')
    parser.add_argument('--head', type=str, help='Whether reconstructing brain data.', default='False')
    parser.add_argument('--ct', type=str, help='Whether reconstructing brain data.', default='False')
    parser.add_argument('--ct_head', type=str, help='Whether reconstructing brain data.', default='False')
    parser.add_argument('--model', type=str, help='which config file to use', required=True, choices=['unet', 'tv'])
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='gaussian1d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d', 'poisson', 'radial'])
    parser.add_argument('--acc_factor', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=4)
    parser.add_argument('--problem', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='lim_angle',
                        choices=['few_view', 'lim_angle'])
    parser.add_argument('--angle', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=1)
    parser.add_argument('--num_views', type=int, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=180)
    parser.add_argument('--center_fraction', type=float, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=0.08)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--N', type=int, help='Number of iterations for score-POCS sampling', default=2000)
    parser.add_argument('--m', type=int, help='Number of corrector step per single predictor step.'
                                              'It is advised not to change this default value.', default=1)
    return parser


if __name__ == "__main__":
    main()