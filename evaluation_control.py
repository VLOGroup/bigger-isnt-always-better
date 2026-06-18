import time
import numpy as np
import logging

from sympy import python
import mydata
import torch
from utils import (fft2, ifft2, get_mask,
                   SSIM, nmse, psnr, get_radial_mask,
                   AsMatrix, inner, Div, Grad, apgd, CharbTV,
                   unet_model, unet_normalize, call_unet)
import imageio.v3 as iio
from physics.model import get_measurement_model
from utils import SSIM, nmse, psnr
from torchvision.transforms import v2
from pathlib import Path
import matplotlib.pyplot as plt

from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)

@torch.no_grad()
def estimate_Lipschitz(A, AT, x_shape, W=None, iters=30, tol=1e-6, device=None, dtype=torch.float32):
    """
    Estimate L = ||A||^2 for f(x) = 1/2 || W^{1/2}(A x)||^2 (if W is None, unweighted LS).
    A  : callable, A(x) -> y
    AT : callable, AT(y) -> x
    W  : callable or None, W(y) -> weighted y (apply W, not sqrt(W))
    x_shape: shape of the image tensor (e.g., (1,1,H,W))
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Start with a random unit vector
    v = torch.randn(*x_shape, device=device, dtype=dtype)
    v = v / (v.flatten().norm() + 1e-12)

    lam_old = 0.0
    for k in range(iters):
        y = A(v)                       # forward
        if W is not None:
            y = W(y)                   # apply weights (not sqrt)
        w = AT(y)                      # backproject => (Aᵀ W A) v  or (Aᵀ A) v
        # Rayleigh quotient gives the eigenvalue estimate
        lam = float((v.flatten() @ w.flatten()) / (v.flatten() @ v.flatten() + 1e-12))
        # Normalize for the next iteration
        v = w / (w.flatten().norm() + 1e-12)

        if abs(lam - lam_old) <= tol * max(1.0, abs(lam)):
            break
        lam_old = lam

    L = max(lam, 1e-12)  # guard against tiny values
    return L  # this is ||A||^2 for unweighted, ||W^{1/2} A||^2 for weighted

def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std

def evaluate(model_config, data_config, evaluation_config, sample_config, savedir, device, logger=None):
  if logger is None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

  savedir.mkdir(parents=True, exist_ok=True)

  if model_config['model']['name'] == 'unet':
    model = unet_model(model_config['path'])
    model.eval()
    logger.info(f'UNet with {sum(p.numel() for p in model.parameters())} parameters')
  elif model_config['model']['name'] == 'tv':
    model = CharbTV(eps=model_config['model']['eps'])
  else:
    raise NotImplementedError(f"Model {model_config['model']['name']} unknown.")
  
  # Get the respective dataloader
  if data_config['dataset'] == 'fastmri_knee':
    _, test_dl = mydata.create_dataloader(Path(data_config['root']), fat_suppression=data_config['fat_suppression'])
  elif data_config['dataset'] == 'fastmri_brain':
    test_dl = mydata.create_brain_dataloader(Path(data_config['root']))
  elif data_config['dataset'] == 'ct' or data_config['dataset'] == 'ct_thorax':
    _, test_dl = mydata.create_ct_dataloader(Path(data_config['root']), 320, batch_size=1)
  elif data_config['dataset'] == 'ct_head':
    test_dl = mydata.create_ct_head_dataloader(Path(data_config['root']), 320, batch_size=1)
  else:
    raise NotImplementedError(f"Dataset {data_config['dataset']} unknown.")
  
  num_data = len(test_dl.dataset)
  logger.info(f'Number of evaluation data available: {num_data}')

  measurement_model = get_measurement_model(evaluation_config['measurement'])
  if evaluation_config['measurement']['methodology'] == 'mri':
    iio.imwrite(savedir / 'mask.png', np.clip(measurement_model.mask.squeeze().detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))
  # Evaluation metrics
  ssim = SSIM().cuda()

  num_recos = min(len(test_dl), evaluation_config['num_samples'])
  if model_config['model']['name'] == 'unet':
    psnr_values = np.zeros(num_recos)
    ssim_values = np.zeros(num_recos)
    nmse_values = np.zeros(num_recos)
    prediction_times = np.zeros(num_recos)
  else:
    if evaluation_config['measurement']['methodology'] == 'mri':
      lamdas = [10, 50, 100, 250, 500, 750, 1000, 2500, 5000, 10_000]  # , 1750, 2000, 2500, 3000, 5000, 7500, 10000, 25000, 50000, 100000, 200000]
    elif evaluation_config['measurement']['methodology'] == 'ct':
      lamdas = [0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100, 250]
    else:
      raise NotImplementedError(f"Measurement methodology {evaluation_config['measurement']['methodology']} unknown.")
    psnr_values = np.zeros((num_recos, len(lamdas)))
    ssim_values = np.zeros((num_recos, len(lamdas)))
    nmse_values = np.zeros((num_recos, len(lamdas)))
    prediction_times = np.zeros((num_recos, len(lamdas)))

  if (savedir / 'psnr_values.csv').is_file():
    psnr_values = np.loadtxt(savedir / 'psnr_values.csv', delimiter=',')
    ssim_values = np.loadtxt(savedir / 'ssim_values.csv', delimiter=',')
    nmse_values = np.loadtxt(savedir / 'nmse_values.csv', delimiter=',')
    prediction_times = np.loadtxt(savedir / 'prediction_times.csv', delimiter=',')

  Lip = estimate_Lipschitz(measurement_model.A, measurement_model.AT, (1, 1, 320, 320), device=device, iters=1000)
  print(f'L = {Lip}')

  
  try:
    for i, img in enumerate(test_dl):
      if i == evaluation_config['num_samples']:
        break
      if (nmse_values[i] != 0).all():
        logger.info(f"Sample {i+1}/{num_recos} already reconstructed.")
        continue
      logger.info(f"Reconstructing sample {i+1}/{num_recos}")
      img = img.view(1, 1, 320, 320).to(device).clamp(min=1e-6)
      under_kspace = measurement_model.A(img)
      under_img = measurement_model.AT(under_kspace) / measurement_model.norm_constant.clip(min=1e-6)

      under_img_scaled = under_img.clone()
      under_img_scaled = under_img_scaled - under_img_scaled.min()
      under_img_scaled = under_img_scaled / under_img_scaled.max()

      # if img.max() > 1e-6:
      #   img = img - img.min()
      #   img = img / img.max()

      iio.imwrite(savedir /  f'sample_{i}_label.png', np.clip(img.squeeze().detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))
      iio.imwrite(savedir /  f'sample_{i}_under.png', np.clip(under_img.squeeze().detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))
      iio.imwrite(savedir /  f'sample_{i}_under_scaled.png', np.clip(under_img_scaled.squeeze().detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))
      plt.imsave(savedir / 'norm_constant.png', measurement_model.norm_constant.squeeze().detach().cpu().numpy(), cmap='inferno')

      tic = time.time()

      if model_config['model']['name'] == 'unet':
        with torch.no_grad():
          input, mean, std = normalize_instance(under_img, eps=1e-11)
          input = input.clamp(-6, 6)
          recon = call_unet(model, under_img if evaluation_config['measurement']['methodology'] == 'mri' else under_img)
          # recon = std * recon + mean
          print(i, img.min().item(), img.max().item(), recon.min().item(), recon.max().item())

          if img.max() > 1e-6 and recon.max() > 1e-6:
            recon -= recon.min()
            recon /= recon.max()
            img -= img.min()
            img /= img.max()

          prediction_times[i] = time.time() - tic
          psnr_values[i] = psnr(recon, img)
          ssim_values[i] = ssim(recon, img)
          nmse_values[i] = nmse(recon, img)

          diff = torch.abs(recon - img).squeeze().detach().cpu().numpy()
          recon = recon.squeeze().detach().cpu().numpy()
          iio.imwrite(savedir /  f'sample_{i}_recon.png', np.clip(recon * 255, 0, 255).astype(np.uint8))
          plt.imsave(savedir / f'sample_{i}_diff.png', diff, vmin=0, vmax=0.15, cmap='inferno')

      else:
        for l, lam in enumerate(lamdas):
          def f_nabla(x):
            dterm = lam * ((measurement_model.A(x) - under_kspace).abs() ** 2).sum((1, 2, 3)) / 2
            nabla_dterm = lam * measurement_model.AT(measurement_model.A(x) - under_kspace)
            reg, nabla_reg = model.grad(x)
            return dterm[:, None, None, None] + reg, nabla_dterm + nabla_reg
          recon = apgd(under_img, f_nabla, lambda x: f_nabla(x)[0], lambda x, _: x, max_iter=200, gamma=0, L_init=Lip)
          prediction_times[i, l] = time.time() - tic
          # recon /= recon.amax((1, 2, 3), keepdim=True).clip(min=1e-6)
          psnr_values[i, l] = psnr(recon, img)
          ssim_values[i, l] = ssim(recon, img)
          nmse_values[i, l] = nmse(recon, img)

          diff = torch.abs(recon - img).squeeze().detach().cpu().numpy()
          recon = recon.squeeze().detach().cpu().numpy()
          iio.imwrite(savedir /  f'sample_{i}_recon_lam_{lam}.png', np.clip(recon * 255, 0, 255).astype(np.uint8))
          plt.imsave(savedir / f'sample_{i}_diff_lam_{lam}.png', diff, vmin=0, vmax=0.15, cmap='inferno')

      np.savetxt(savedir / 'prediction_times.csv', prediction_times, delimiter=',')
      np.savetxt(savedir / 'psnr_values.csv', psnr_values, delimiter=',')
      np.savetxt(savedir / 'ssim_values.csv', ssim_values, delimiter=',')
      np.savetxt(savedir / 'nmse_values.csv', nmse_values, delimiter=',')

  except Exception as e:
    logger.error("An error occurred during evaluation: %s", e)
    raise

  finally:
    np.savetxt(savedir / 'prediction_times.csv', prediction_times, delimiter=',')
    np.savetxt(savedir / 'psnr_values.csv', psnr_values, delimiter=',')
    np.savetxt(savedir / 'ssim_values.csv', ssim_values, delimiter=',')
    np.savetxt(savedir / 'nmse_values.csv', nmse_values, delimiter=',')

  return
