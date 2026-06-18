# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools
import time

import torch
import numpy as np
import abc

import matplotlib.pyplot as plt
import functools
from utils import fft2, ifft2, clear, fft2_m, ifft2_m, root_sum_of_squares, psnr, nmse, SSIM
from tqdm import tqdm, trange
from models import utils as mutils
import imageio.v3 as iio

_CORRECTORS = {}
_PREDICTORS = {}

ssim = SSIM().cuda()


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, continuous, eps, device, depth_param=False):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config['method']
  predictor = get_predictor(config['predictor'].lower())
  corrector = get_corrector(config['corrector'].lower())
  sampling_fn = get_pc_sampler(sde=sde,
                               shape=shape,
                               predictor=predictor,
                               corrector=corrector,
                               inverse_scaler=inverse_scaler,
                               snr=config['snr'],
                               n_steps=config['n_steps_each'],
                               probability_flow=config['probability_flow'],
                               continuous=continuous,
                               denoise=config['noise_removal'],
                               eps=eps,
                               device=device,
                               depth_param=depth_param)
  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t, d=0):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t, d=0):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t, d=0):
    f, G = self.rsde.discretize(x, t, d)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)

  def update_fn(self, x, t, d=0):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    alpha = torch.ones_like(t)

    for i in range(n_steps):
      if not (d == 0 or d is None):
        grad = score_fn(x, t, d)
      else:
        grad = score_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean


def shared_predictor_update_fn(x, t, d, sde, model, predictor, probability_flow, continuous, depth_param=False):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous, depth_param=depth_param)
  predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t, d)


def shared_corrector_update_fn(x, t, d, sde, model, corrector, continuous, snr, n_steps, depth_param=False):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous, depth_param=depth_param)
  corrector_obj = corrector(sde, score_fn, snr, n_steps)
  fn = corrector_obj.update_fn(x, t, d)
  return fn


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda', depth_param=False):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous,
                                          depth_param=depth_param)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps,
                                          depth_param=depth_param)

  def pc_sampler(model, d=None):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

      time_corrector_tot = 0
      time_predictor_tot = 0
      for i in trange(sde.N, desc='PC Sampling'):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        tic_corrector = time.time()
        x, x_mean = corrector_update_fn(x, vec_t, d, model=model)
        time_corrector_tot += time.time() - tic_corrector
        tic_predictor = time.time()
        x, x_mean = predictor_update_fn(x, vec_t, d, model=model)
        time_predictor_tot += time.time() - tic_predictor

      return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return pc_sampler


def get_conditional_pc_sampler(sde, predictor, corrector, inverse_scaler, snr,
                              n_steps=1, probability_flow=False, continuous=False,
                              denoise=True, eps=1e-5, save_progress=False, save_root=None, lambda_value=1, depth_param=False):
  """Create a PC sampler for solving compressed sensing problems as in MRI reconstruction.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for the corrector.
    n_steps: An integer. The number of corrector steps per update of the corrector.
    continuous: `True` indicates that the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

  Returns:
    A CS solver function.
  """
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous,
                                          depth_param=depth_param)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps,
                                          depth_param=depth_param)

  def data_fidelity(measurement_model, x, x_mean, Fy, lambda_value=1):
      """
      Data fidelity operation for Fourier CS
      x: Current aliased img
      Fy: k-space measurement data (masked)
      """
      x = x + 2. * lambda_value * measurement_model.AT(Fy - measurement_model.A(x)) / measurement_model.norm_constant
      x_mean = x_mean + 2. * lambda_value * measurement_model.AT(Fy - measurement_model.A(x_mean)) / measurement_model.norm_constant
      return x, x_mean

  def get_fouriercs_update_fn(update_fn, lambda_value):
    """Modify the update function of predictor & corrector to incorporate data information."""

    def fouriercs_update_fn(model, data, measurement_model, x, t, d, Fy=None):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        x, x_mean = update_fn(x, vec_t, d, model=model)
        x, x_mean = data_fidelity(measurement_model, x, x_mean, Fy, lambda_value)
        return x, x_mean

    return fouriercs_update_fn

  projector_fouriercs_update_fn = get_fouriercs_update_fn(predictor_update_fn, lambda_value)
  corrector_fouriercs_update_fn = get_fouriercs_update_fn(corrector_update_fn, lambda_value)

  def pc_fouriercs(model, data, measurement_model, Fy=None, d=None, savedir=None, label=None):
    with torch.no_grad():
      # Initial sample
      # x = torch.real(ifft2(Fy + fft2(sde.prior_sampling(data.shape).to(data.device)) * (1. - mask)))
      if not depth_param:
        d=0
      x = sde.prior_sampling(data.shape).to(data.device)
      x = x # + measurement_model.AT(Fy - measurement_model.A(x))
      timesteps = torch.linspace(sde.T, eps, sde.N)
      for i in tqdm(range(sde.N), total=sde.N, desc='PC Sampling'):
        t = timesteps[i]
        x, x_mean = corrector_fouriercs_update_fn(model, data, measurement_model, x, t, d, Fy=Fy)
        x, x_mean = projector_fouriercs_update_fn(model, data, measurement_model, x, t, d, Fy=Fy)
        if save_progress and i >= 300 and i % 100 == 0:
          plt.imsave(save_root / f'step{i}.png', clear(x_mean), cmap='gray')
          
      return inverse_scaler(x_mean if denoise else x)

  return pc_fouriercs
  

def get_conditional_ald_sampler(sde, inverse_scaler, continuous=False, eps=1e-5, lambda_value=1):
  """Create a PC sampler for solving compressed sensing problems as in MRI reconstruction.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for the corrector.
    n_steps: An integer. The number of corrector steps per update of the corrector.
    continuous: `True` indicates that the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

  Returns:
    A CS solver function.
  """

  

  def ald_sampler(model, data, measurement_model, Fy=None, d=None, savedir=None, label=None):
    with torch.no_grad():
      # Initial sample
      if d is not None:
        score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous, depth_param=True)
      else:
        score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
      x = torch.randn_like(data)
      timesteps = torch.linspace(sde.T, eps, sde.N)

      if sde.N == 2000:
        N_start = 1111
      elif sde.N == 1000:
        N_start = 555
      elif sde.N == 500:
        N_start = 280
      elif sde.N == 250:
        N_start = 140
      else:
        N_start = 0

      for i in trange(N_start, sde.N, initial=N_start, total=sde.N, desc='ALD Sampling'):

        t = timesteps[i]

        step_size = 5e-5 * (sde.discrete_sigmas[sde.N - i - 1] / sde.sigma_min) ** 2
        for _ in range(3):
          noise = torch.randn_like(x) * torch.sqrt(2 * step_size)
          if d is not None:
            p_grad = score_fn(x, torch.ones(data.shape[0], device=data.device) * t, d)
          else:
            p_grad = score_fn(x, torch.ones(data.shape[0], device=data.device) * t)
          meas_grad = measurement_model.AT(measurement_model.A(x) - Fy)
          meas_grad /= torch.norm(meas_grad)
          meas_grad *= lambda_value * 5 * torch.norm(p_grad)

          x = x + step_size * (p_grad - meas_grad) + noise

      return inverse_scaler(x)

  return ald_sampler


def get_mcg_solver(sde, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False, weight=1.0,
                   denoise=True, eps=1e-5, radon=None, radon_all=None, save_progress=False, save_root=None,
                   lamb_schedule=None, mask=None, measurement_noise=False):
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  # def _A(x):
  #     return radon.A(x)

  # def _AT(sinogram):
  #     return radon.AT(sinogram)

  # def _AINV(sinogram):
  #     return radon.A_dagger(sinogram)

  # def _A_all(x):
  #     return radon_all.A(x)

  # def _AINV_all(sinogram):
  #     return radon_all.A_dagger(sinogram)

  def get_update_fn(update_fn):
    def radon_update_fn(model, data, x, t, d):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        x, _, _ = update_fn(x, vec_t, 0, model=model)
        return x

    return radon_update_fn

  def get_corrector_update_fn(update_fn):
    def radon_update_fn(model, data, measurement_model, x, t, measurement=None, i=None):
      vec_t = torch.ones(data.shape[0], device=data.device) * t

      # mn True
      if measurement_noise:
        measurement_mean, std = sde.marginal_prob(measurement, vec_t)
        measurement = measurement_mean + torch.randn_like(measurement) * std[:, None, None, None]

      # input to the score function
      x = x.requires_grad_()
      x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

      lamb = lamb_schedule.get_current_lambda(i)

      # x0 hat estimation
      _, bt = sde.marginal_prob(x, vec_t)
      hatx0 = x + (bt ** 2) * score

      # MCG method
      # norm = torch.linalg.norm(_AINV(measurement - _A(hatx0)))
      norm = torch.norm(measurement_model.A_dagger(measurement - measurement_model.A(hatx0)))
      norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
      norm_grad *= weight
      norm_grad = measurement_model.A_dagger_all(measurement_model.A_all(norm_grad) * (1. - mask))

      x_next = x_next + lamb * measurement_model.AT(measurement - measurement_model.A(x_next)) / measurement_model.norm_constant - norm_grad
      x_next = x_next.detach()
      return x_next

    return radon_update_fn

  predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
  corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

  def pc_radon(model, data, measurement_model, Fy=None):
    x = sde.prior_sampling(data.shape).to(data.device)
    timesteps = torch.linspace(sde.T, eps, sde.N)
    for i in tqdm(range(sde.N)):
      t = timesteps[i]
      x = predictor_denoise_update_fn(model, data, measurement_model, x, t)
      x = corrector_radon_update_fn(model, data, measurement_model, x, t, measurement=Fy, i=i)
      if save_progress:
        if (i % 100) == 0:
          plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x), cmap='gray')

    return inverse_scaler(x if denoise else x)

  return pc_radon



def get_dps_sde_sampler(sde, inverse_scaler, continuous=False, eps=1e-5, lambda_value=1., use_yt=False, use_N_start=True):

  def dps_sampler(model, data, measurement_model, Fy=None, d=None, savedir=None, label=None):
    torch.manual_seed(42)
    if savedir is not None:
      savedir.mkdir(parents=True, exist_ok=True)
    if d is not None:
      score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous, depth_param=True)
    else:
      score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    x = sde.prior_sampling(data.shape).to(data.device)
    x = x + measurement_model.AT(Fy - measurement_model.A(x))/measurement_model.norm_constant
    timesteps = torch.linspace(sde.T, eps, sde.N)

    psnr_values_x = []
    ssim_values_x = []
    nmse_values_x = []

    psnr_values_x0_hat = []
    ssim_values_x0_hat = []
    nmse_values_x0_hat = []
    iters = []

    for i in tqdm(range(sde.N), total=sde.N, desc='DPS SDE Sampling'):
      t = timesteps[i]
      sigma = sde.discrete_sigmas[sde.N - i - 1]
      x.requires_grad_()
      if d is not None:
        score = score_fn(x, torch.ones(data.shape[0], device=data.device) * t, d)
      else:
        score = score_fn(x, torch.ones(data.shape[0], device=data.device) * t)
      x0_hat = x + sigma ** 2 * score
      if use_yt:
        y_t = Fy + sigma * torch.randn_like(Fy)
        norm = torch.norm(y_t - measurement_model.A(x0_hat)) ** 2
      else:
        norm = torch.norm(Fy - measurement_model.A(x0_hat)) ** 2
      likelihood_score = torch.autograd.grad(outputs=norm, inputs=x)[0] / measurement_model.norm_constant  # 0.5 * torch.autograd.grad(outputs=norm, inputs=x)[0] / measurement_model.norm_constant
      posterior_score = score - lambda_value/(sigma ** 2) * likelihood_score

      z = torch.randn_like(x)
      _, G = sde.discretize(x, t, 0)
      rev_f = - G ** 2 * posterior_score
      x = x - rev_f + G * z
      x.detach()
      if savedir is not None:
        if i % 25 == 0 or (i > sde.N - 50 and i % 5 == 0) or i == sde.N - 1:
          iio.imwrite(savedir / f'x0_hat_{i}.png', np.clip(clear(x0_hat) * 255, 0, 255).astype(np.uint8))
          iio.imwrite(savedir / f'x_{i}.png', np.clip(clear(x) * 255, 0, 255).astype(np.uint8))
          psnr_values_x.append(psnr(x, label).item())
          ssim_values_x.append(ssim(x, label).item())
          nmse_values_x.append(nmse(x, label).item())
          psnr_values_x0_hat.append(psnr(x0_hat, label).item())
          ssim_values_x0_hat.append(ssim(x0_hat, label).item())
          nmse_values_x0_hat.append(nmse(x0_hat, label).item())
          iters.append(i)

    if savedir is not None:
      np.savetxt(savedir / 'psnr_values_x.txt', np.array(psnr_values_x))
      np.savetxt(savedir / 'ssim_values_x.txt', np.array(ssim_values_x))
      np.savetxt(savedir / 'nmse_values_x.txt', np.array(nmse_values_x))
      np.savetxt(savedir / 'psnr_values_x0_hat.txt', np.array(psnr_values_x0_hat))
      np.savetxt(savedir / 'ssim_values_x0_hat.txt', np.array(ssim_values_x0_hat))
      np.savetxt(savedir / 'nmse_values_x0_hat.txt', np.array(nmse_values_x0_hat))

      fig, ax1 = plt.subplots(figsize=(12, 8))
      ax1.plot(iters, psnr_values_x, label=r'PSNR $x$', c='C0')
      ax1.plot(iters, psnr_values_x0_hat, label=r'PSNR $\hat{x}_0$', c='C0', linestyle='--')
      ax1.grid(True, axis='x')
      ax1.set_xlabel('Iteration')
      ax1.set_ylabel('PSNR in dB')

      ax2 = ax1.twinx()
      ax2.plot(iters, ssim_values_x, label=r'SSIM $x$', c='C1')
      ax2.plot(iters, ssim_values_x0_hat, label=r'SSIM $\hat{x}_0$', c='C1', linestyle='--')
      ax2.plot(iters, nmse_values_x, label=r'NMSE $x$', c='C2')
      ax2.plot(iters, nmse_values_x0_hat, label=r'NMSE $\hat{x}_0$', c='C2', linestyle='--')
      ax2.set_ylim(0, 1.2)
      ax2.grid(True, axis='both')
      ax2.set_ylabel('SSIM and NMSE in a.u.')
      fig.legend(loc='lower left')
      plt.title('Metrics over Iterations')
      fig.tight_layout()

      plt.savefig(savedir / 'metrics.png')
      plt.close('all')

    return inverse_scaler(x0_hat)

  return dps_sampler

def get_dps_sampler(sde, zeta, inverse_scaler, continuous=False, eps=1e-5, use_yt=False, use_N_start=True):
  def dps_sampler(model, data, measurement_model, Fy=None, d=None, savedir=None, label=None):
    torch.manual_seed(42)
    if savedir is not None:
      savedir.mkdir(parents=True, exist_ok=True)
    if d is not None:
      score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous, depth_param=True)
    else:
      score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    x = sde.prior_sampling(data.shape).to(data.device)
    x = torch.randn_like(data)
    x = x + measurement_model.AT(Fy - measurement_model.A(x))/measurement_model.norm_constant
    timesteps = torch.linspace(sde.T, eps, sde.N)

    psnr_values_x = []
    ssim_values_x = []
    nmse_values_x = []

    psnr_values_x0_hat = []
    ssim_values_x0_hat = []
    nmse_values_x0_hat = []
    iters = []
    if use_N_start:
      if sde.N == 2000:
        N_start = 1111
      elif sde.N == 1000:
        N_start = 555
      elif sde.N == 500:
        N_start = 280
      elif sde.N == 250:
        N_start = 140
      else:
        N_start = 0
    else:
      N_start = 0

    for i in trange(N_start, sde.N, initial=N_start, total=sde.N, desc='DPS Sampling'):
      t = timesteps[i]
      sigma = sde.discrete_sigmas[sde.N - i - 1]
      x.requires_grad_()
      if d is not None:
        score = score_fn(x, torch.ones(data.shape[0], device=data.device) * t, d)
      else:
        score = score_fn(x, torch.ones(data.shape[0], device=data.device) * t)
      x0_hat = x + sigma ** 2 * score
      if use_yt:
        y_t = Fy + sigma * torch.randn_like(Fy)
        norm = torch.norm(y_t - measurement_model.A(x0_hat)) ** 2
      else:
        norm = torch.norm(Fy - measurement_model.A(x0_hat)) ** 2

      z = torch.randn_like(x)
      grad = zeta * torch.autograd.grad(outputs=norm, inputs=x)[0] / measurement_model.norm_constant
      if i == sde.N-1:
        x = x0_hat - grad
      else:
        x = x0_hat + sde.discrete_sigmas[sde.N - i - 2] * z - grad

      if savedir is not None:
        if i % 5 == 0 or i == sde.N - 1 or i < 5:  # i % 25 == 0 or (i > sde.N - 50 and i % 5 == 0) or i == sde.N - 1:
          iio.imwrite(savedir / f'x0_hat_{i}.png', np.clip(clear(x0_hat) * 255, 0, 255).astype(np.uint8))
          iio.imwrite(savedir / f'x_{i}.png', np.clip(clear(x) * 255, 0, 255).astype(np.uint8))
          psnr_values_x.append(psnr(x, label).item())
          ssim_values_x.append(ssim(x, label).item())
          nmse_values_x.append(nmse(x, label).item())
          psnr_values_x0_hat.append(psnr(x0_hat, label).item())
          ssim_values_x0_hat.append(ssim(x0_hat, label).item())
          nmse_values_x0_hat.append(nmse(x0_hat, label).item())
          iters.append(i)

    return inverse_scaler(x)

  return dps_sampler

def get_diffpir_sampler(sde, inverse_scaler, continuous=False, eps=1e-5, lambda_value=1.):

  def diffpir_sampler(model, data, measurement_model, Fy=None):
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    x = sde.prior_sampling(data.shape).to(data.device)
    x = x + measurement_model.AT(Fy - measurement_model.A(x))/measurement_model.norm_constant
    timesteps = torch.linspace(sde.T, eps, sde.N)

    norms = []
    likelihood_scores = []
    posterior_scores = []
    scores = []


    # to convert to DDPM:
    # betas = torch.from_numpy(np.linspace(0.0001, 0.02, 2000, dtype=np.float32))
    # alphas = 1. - betas
    N = 150
    alphas = sde.discrete_sigmas[:N]
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)
    sigmas = torch.sqrt((1 - alphas) / alphas)
    zeta = torch.tensor([0.1]).to('cuda:0')

    x = torch.randn_like(data)

    for i in tqdm(range(N), total=N, desc='DiffPIR Sampling'):  # tqdm(range(sde.N), total=sde.N):
      t = timesteps[i]
      sigma = sde.discrete_sigmas[N - i - 1]
      n = N - i - 1
      # sigma = reduced_alpha_cumprod[n]
      score = model(x * torch.sqrt(1 + sigma ** 2), torch.ones(data.shape[0], device=data.device) * sigma)
      x_0 = (x + (1 - alphas_cumprod[n]) * score) / sqrt_alphas_cumprod[n]
      x_0.requires_grad_()
      norm = torch.norm(Fy - measurement_model.A(x_0)) ** 2
      x_0_hat = x_0 - reduced_alpha_cumprod[n] / 2 * torch.autograd.grad(outputs=norm, inputs=x_0)[0] / measurement_model.norm_constant
      epsilon_hat = (x - sqrt_alphas_cumprod[n] * x_0_hat) / sqrt_1m_alphas_cumprod[n]
      x_0= sqrt_alphas_cumprod[n - 1] * x_0_hat + sqrt_1m_alphas_cumprod[n - 1] * (torch.sqrt(1 - zeta) * epsilon_hat + torch.sqrt(zeta) * torch.randn_like(x)) 

      x.detach()

      if i % 100 == 0 or i == sde.N - 1:
        plt.imsave(f'score_{i}.png', clear(score), cmap='gray')
        plt.imsave(f'x0_{i}.png', clear(x_0), cmap='gray')
        plt.imsave(f'x_{i}.png', clear(x), cmap='gray')
        plt.imsave(f'x0_hat{i}.png', clear(x_0_hat), cmap='gray')

    return inverse_scaler(x)

  return diffpir_sampler