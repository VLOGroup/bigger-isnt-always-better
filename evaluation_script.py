import time
import numpy as np
import logging
from models import mrncsn, msncsn, ncsnpp, mrncsn_structured
import losses
import sampling, sampling_vp
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import mydata
import sde_lib
import torch
from torch.utils import tensorboard
from utils import save_checkpoint, restore_checkpoint
import imageio.v3 as iio
from physics.model import get_measurement_model
from utils import SSIM, nmse, psnr
from torchvision.transforms import v2
from pathlib import Path
import matplotlib.pyplot as plt

def evaluate(model_config, data_config, evaluation_config, sample_config, savedir, device, logger=None):
  if logger is None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

  savedir.mkdir(parents=True, exist_ok=True)


  # Initialize model.
  score_model = mutils.create_model(model_config)
  score_model = score_model.to(device)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=model_config['model']['ema_rate'])
  state = dict(model=score_model, ema=ema, step=0)
  state = restore_checkpoint(model_config['path'], state, device)
  # ema.copy_to(score_model.parameters())
  print(f'{state["step"]=}')

  logger.info(f'Model Architecture:\n{score_model.all_modules}')
  
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

  # Create data normalizer and its inverse
  scaler = mydata.get_data_scaler(data_config)
  inverse_scaler = mydata.get_data_inverse_scaler(data_config)

  # Setup SDEs
  if model_config['sde']['sde'].lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=model_config['model']['beta_min'], beta_max=model_config['model']['beta_max'], N=sample_config['num_scales'])
    sampling_eps = 1e-3
  elif model_config['sde']['sde'].lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=model_config['model']['beta_min'], beta_max=model_config['model']['beta_max'], N=sample_config['num_scales'])
    sampling_eps = 1e-3
  elif model_config['sde']['sde'].lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=model_config['model']['sigma_min'], sigma_max=model_config['model']['sigma_max'], N=sample_config['num_scales'])
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {model_config['sde']['sde']} unknown.")

  measurement_model = get_measurement_model(evaluation_config['measurement'])
  if evaluation_config['measurement']['methodology'] == 'mri':
    iio.imwrite(savedir / 'mask.png', np.clip(measurement_model.mask.squeeze().detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))

  if sample_config['method'] == 'pc':
    predictor = sampling.ReverseDiffusionPredictor
    corrector = sampling.LangevinCorrector
    snr = 0.16
    sampler = sampling.get_conditional_pc_sampler(sde, predictor, corrector, inverse_scaler, snr,
                                              n_steps=1, probability_flow=sample_config['probability_flow'], continuous=model_config['sde']['continuous'],
                                              denoise=True, eps=sampling_eps, save_progress=False, save_root=None, lambda_value=1,
                                              depth_param=(model_config['model']['name'] == 'msncsn' or model_config['model']['name'] == 'mrncsn' or model_config['model']['name'] == 'mrncsn_structured'))
  elif sample_config['method'] == 'ald':
    sampler = sampling.get_conditional_ald_sampler(sde, inverse_scaler, continuous=model_config['sde']['continuous'], eps=sampling_eps, lambda_value=1)
  elif sample_config['method'] == 'dps':
    if model_config['sde']['sde'].lower() == 'vesde':
      sampler = sampling.get_dps_sampler(sde, evaluation_config['zeta'], inverse_scaler, continuous=model_config['sde']['continuous'], eps=sampling_eps, use_yt=sample_config['use_yt'])
    elif model_config['sde']['sde'].lower() == 'vpsde':
      sampler = sampling_vp.get_dps_sampler(sde, inverse_scaler, continuous=model_config['sde']['continuous'], eps=sampling_eps, lambda_value=1., use_yt=sample_config['use_yt'], savedir=savedir / 'intermediates')
    else:
      raise NotImplementedError(f"SDE {model_config['sde']['sde']} unknown.")
  elif sample_config['method'] == 'dps_sde':
    sampler = sampling.get_dps_sde_sampler(sde, inverse_scaler, continuous=model_config['sde']['continuous'], eps=sampling_eps, lambda_value=1., use_yt=sample_config['use_yt'])
  elif sample_config['method'] == 'diffpir':
    if model_config['sde']['sde'].lower() == 'vesde':
      sampler = sampling.get_diffpir_sampler(sde, inverse_scaler, continuous=model_config['sde']['continuous'], eps=sampling_eps)
    elif model_config['sde']['sde'].lower() == 'vpsde':
      sampler = sampling_vp.get_diffpir_sampler(sde, inverse_scaler, continuous=model_config['sde']['continuous'], eps=sampling_eps, savedir=savedir / 'intermediates')

  else:
    raise NotImplementedError(f"Sampling method {sample_config['method']} unknown.")
  
  # Evaluation metrics
  ssim = SSIM().cuda()

  num_recos = min(len(test_dl), evaluation_config['num_samples'])
  if model_config['model']['name'] == 'mrncsn' or model_config['model']['name'] == 'msncsn' or model_config['model']['name'] == 'mrncsn_structured':
    psnr_values = np.zeros((num_recos, model_config['model']['num_resolutions']))
    ssim_values = np.zeros((num_recos, model_config['model']['num_resolutions']))
    nmse_values = np.zeros((num_recos, model_config['model']['num_resolutions']))
  else:
    psnr_values = np.zeros(num_recos)
    ssim_values = np.zeros(num_recos)
    nmse_values = np.zeros(num_recos)
  prediction_times = np.zeros(num_recos)

  if (savedir / 'psnr_values.csv').is_file():
    psnr_values = np.loadtxt(savedir / 'psnr_values.csv', delimiter=',')
    ssim_values = np.loadtxt(savedir / 'ssim_values.csv', delimiter=',')
    nmse_values = np.loadtxt(savedir / 'nmse_values.csv', delimiter=',')
    prediction_times = np.loadtxt(savedir / 'prediction_times.csv', delimiter=',')

  print(f'{measurement_model.norm_constant.min()=}, {measurement_model.norm_constant.max()=}')

  
  try:
    for i, img in enumerate(test_dl):
      if i == evaluation_config['num_samples']:
        break
      if (nmse_values[i] != 0).all():
        logger.info(f"Sample {i+1}/{num_recos} already reconstructed.")
        continue
      logger.info(f"Reconstructing sample {i+1}/{num_recos}")
      img = img.view(1, 1, 320, 320).to(device)
      img -= img.min()
      img /= img.max()
      print(img.min(), img.max())
      under_kspace = measurement_model.A(img)
      under_img = measurement_model.AT(under_kspace) / measurement_model.norm_constant.clip(min=1e-6)
      print(under_img.min(), under_img.max())
      x = torch.randn_like(img)
      y = torch.randn_like(under_kspace)
      print(torch.sum(measurement_model.A(x) * y), torch.sum(x * measurement_model.AT(y)))

      under_img_scaled = under_img.clone()
      # under_img_scaled = under_img_scaled - under_img_scaled.min()
      # under_img_scaled = under_img_scaled / under_img_scaled.max()

      iio.imwrite(savedir /  f'sample_{i}_label.png', np.clip(inverse_scaler(img).squeeze().detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))
      iio.imwrite(savedir /  f'sample_{i}_under.png', np.clip(inverse_scaler(under_img).squeeze().detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))
      iio.imwrite(savedir /  f'sample_{i}_under_scaled.png', np.clip(inverse_scaler(under_img_scaled).squeeze().detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))
      plt.imsave(savedir / 'norm_constant.png', measurement_model.norm_constant.squeeze().detach().cpu().numpy(), cmap='inferno')

      if model_config['model']['name'] == 'mrncsn' or model_config['model']['name'] == 'msncsn' or model_config['model']['name'] == 'mrncsn_structured':
        for d in range(model_config['model']['num_resolutions']):
          logger.info(f'd = {d+1}')
          tic = time.time()
          recon = sampler(score_model, under_img, measurement_model, Fy=under_kspace, d=d+1, savedir=savedir / 'intermediates' / f'sample_{i}_d{d}', label=img)

          prediction_times[i] = time.time() - tic
          psnr_values[i, d] = psnr(recon, img)
          ssim_values[i, d] = ssim(recon, img)
          nmse_values[i, d] = nmse(recon, img)

          diff = torch.abs(recon - img).squeeze().detach().cpu().numpy()
          recon = recon.squeeze().detach().cpu().numpy()
          iio.imwrite(savedir /  f'sample_{i}_recon_d{d}.png', np.clip(recon * 255, 0, 255).astype(np.uint8))
          plt.imsave(savedir / f'sample_{i}_diff_d{d}.png', diff, vmin=0, vmax=0.15, cmap='inferno')


      else:
        tic = time.time()
        recon = sampler(score_model, under_img_scaled, measurement_model, Fy=under_kspace, savedir=savedir / 'intermediates' / f'sample_{i}', label=img).clamp(0, 1)

        prediction_times[i] = time.time() - tic
        # img -= img.min()
        # img /= img.max()
        # recon -= recon.min()
        # recon /= recon.max()
        psnr_values[i] = psnr(recon, img)
        ssim_values[i] = ssim(recon, img)
        nmse_values[i] = nmse(recon, img)

        diff = torch.abs(recon - img).squeeze().detach().cpu().numpy()
        recon = recon.squeeze().detach().cpu().numpy()
        iio.imwrite(savedir /  f'sample_{i}_recon.png', np.clip(recon * 255, 0, 255).astype(np.uint8))
        plt.imsave(savedir / f'sample_{i}_diff.png', diff, vmin=0, vmax=0.15, cmap='inferno')

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
