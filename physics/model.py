import numpy as np
import torch
from  physics.mri import MRI
from physics.ct import CT, CT_LA, CTFanBeam
from utils import (fft2, ifft2, get_mask, get_data_scaler, 
                   get_data_inverse_scaler, restore_checkpoint, 
                   SSIM, nmse, psnr, get_radial_mask, get_outer_mask, get_inner_mask)

def get_measurement_model(config):
    img_size = config['img_size']
    if 'seed' in config:
        np.random.seed(config['seed'])
    if config['methodology'] == 'mri':
        if config['mask'] == 'radial':
            mask = get_radial_mask((320, 320), config['num_spokes'], np.pi / config['num_spokes'])
        elif config['mask'] == 'outer':
            mask = get_outer_mask(config['cutout'])
        elif config['mask'] == 'inner':
            mask = get_inner_mask(config['cutout'])
        elif config['mask'] == 'gaussian1d':
            mask = get_mask(torch.zeros([1, 1, img_size, img_size]), img_size, config['batch_size'], type=config['mask'], acc_factor=config['acc_factor'], center_fraction=config['center_fraction'])
        else:
            mask = get_mask(torch.zeros([1, 1, img_size, img_size]), img_size, config['batch_size'], type=config['mask'], acc_factor=config['acc_factor'])
        return MRI(mask)
    elif config['methodology'] == 'ct':
        if config['mask'] == 'sparse_view':
            ct = CT(img_width=img_size, n_views=config['num_views'])
        elif config['mask'] == 'lim_angle':
            ct = CT(img_width=img_size, n_views=config['num_views'], end_angle=config['end_angle'])
        elif config['mask'] == 'fanbeam':
            ct = CTFanBeam(img_width=img_size, n_views=config['num_views'])
        else:
            raise NotImplementedError(f"CT: Undersampling scheme {config['mask']} unknown.")
        return ct
    else:
        raise NotImplementedError(f"Methodology {config['methodology']} unknown.")