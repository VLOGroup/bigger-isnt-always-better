"""Various sampling methods."""
import functools
import time

import torch
import numpy as np
import abc

import matplotlib.pyplot as plt
import functools
from utils import fft2, ifft2, clear, fft2_m, ifft2_m, root_sum_of_squares, psnr, nmse, SSIM
from tqdm import tqdm
from models import utils as mutils


def get_dps_sampler(sde, inverse_scaler, continuous=False, eps=1e-5, lambda_value=1., use_yt=False, savedir=None):

  def dps_sampler(model, data, measurement_model, Fy=None, d=None):
    if savedir is not None:
      savedir.mkdir(parents=True, exist_ok=True)
    print(f'{d=}')
    if d is not None:
      score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous, depth_param=True)
    else:
      score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    x = sde.prior_sampling(data.shape).to(data.device)
    #x = x + measurement_model.AT(Fy - measurement_model.A(x))/measurement_model.norm_constant
    timesteps = torch.linspace(sde.T, eps, sde.N)

    norms = []
    likelihood_scores = []
    posterior_scores = []
    scores = []
    print(len(sde.discrete_betas))
    for i in tqdm(range(sde.N), total=sde.N):
      t = timesteps[i]
      n = sde.N - i - 1
      x.requires_grad_()
      if d is not None:
        score = score_fn(x, torch.ones(data.shape[0], device=data.device) * t, d)
      else:
        score = score_fn(x, torch.ones(data.shape[0], device=data.device) * t)
      x0_hat = (x + (1 - sde.alphas_cumprod[n]) * score) / sde.sqrt_alphas_cumprod[n]

      z = torch.randn_like(x)
      if i == sde.N - 1:
        z = torch.zeros_like(x)
      # x_prime = torch.sqrt(sde.alphas[n]) * (1 - sde.alphas_cumprod[n-1]) / (1 - sde.alphas_cumprod[n]) * x + \
      #           sde.sqrt_alphas_cumprod[n - 1] * sde.discrete_betas[n] / (1 - sde.alphas_cumprod[n]) * x0_hat + torch.sqrt(sde.discrete_betas[n]) * z

      x_prime = (x + sde.discrete_betas[n] * score) / torch.sqrt(sde.alphas[n]) + torch.sqrt(sde.discrete_betas[n]) * z

      if use_yt:
        y_t = Fy # + sigma * torch.randn_like(Fy)
        norm = torch.norm(y_t - measurement_model.A(x0_hat)) ** 2
      else:
        norm = torch.norm(Fy - measurement_model.A(x0_hat)) ** 2
      likelihood_score = torch.autograd.grad(outputs=norm, inputs=x)[0] / norm  # / measurement_model.norm_constant
      x.detach()
      x = x_prime - lambda_value * likelihood_score
      
      # x = (x + sde.discrete_betas[n] * score) / torch.sqrt(sde.alphas[n]) + torch.sqrt(sde.discrete_betas[n]) * z

      if savedir is not None:
        if i % 100 == 0 or i == sde.N - 1:
          plt.imsave(savedir / f'score_{i}.png', clear(score), cmap='gray')
          plt.imsave(savedir / f'x0_hat_{i}.png', clear(x0_hat), cmap='gray')
          plt.imsave(savedir / f'x_{i}.png', clear(x), cmap='gray')
          plt.imsave(savedir / f'likelihood_score_{i}.png', clear(likelihood_score), cmap='gray')

      # Collect values for visualization
    #   norms.append(norm.item())
    #   likelihood_scores.append(torch.norm(likelihood_score).item())
    #   posterior_scores.append(torch.norm(x).item())
    #   scores.append(torch.norm(score).item())

    # # Plot the collected values with two y-axes
    # fig, ax1 = plt.subplots(figsize=(12, 8))

    # ax1.set_xlabel('Iteration')
    # ax1.set_ylabel('Norm', color='tab:blue')
    # ax1.plot(norms, label='Norm', color='tab:blue')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Values', color='tab:orange')
    # ax2.plot(likelihood_scores, label='Likelihood Score Norm', color='tab:orange')
    # ax2.plot(posterior_scores, label='x Norm', color='tab:green')
    # ax2.plot(scores, label='Score Norm', color='tab:red')
    # ax2.tick_params(axis='y', labelcolor='tab:orange')

    # fig.tight_layout()
    # fig.legend(loc='upper right')
    # plt.title('Values over Iterations')
    # plt.savefig(savedir / 'values_over_iterations.png')
    # plt.close()

    return inverse_scaler(x0_hat)

  return dps_sampler

def get_diffpir_sampler(sde, inverse_scaler, continuous=False, eps=1e-5, lambda_value=1., d=None, savedir=None):

  def diffpir_sampler(model, data, measurement_model, Fy=None, d=None):
    model.eval()
    if savedir is not None:
      savedir.mkdir(parents=True, exist_ok=True)
    print(f'{d=}')
    if d is not None:
      score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous, depth_param=True)
    else:
      score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    x = sde.prior_sampling(data.shape).to(data.device)
    timesteps = torch.linspace(sde.T, eps, sde.N)

    norms = []
    x0t_hats = []
    x0ts = []
    scores = []
    eps_hats = []
    xs = []
    etas = []
    reduced_alpha_cumprod = sde.sqrt_1m_alphas_cumprod / sde.sqrt_alphas_cumprod
    print(f'{sde.sqrt_1m_alphas_cumprod=}')
    print(f'{sde.sqrt_alphas_cumprod=}')
    print(f'{reduced_alpha_cumprod=}')


    # to convert to DDPM:
    # betas = torch.from_numpy(np.linspace(0.0001, 0.02, 2000, dtype=np.float32))
    # alphas = 1. - betas
    zeta = torch.tensor([1]).to('cuda:0')

    x = torch.randn_like(data)
    
    for i in tqdm(range(sde.N), total=sde.N):
      t = timesteps[i]
      n = sde.N - i - 1
      x.requires_grad_()
      if d is not None:
        score = score_fn(x, torch.ones(data.shape[0], device=data.device) * t, d).detach()
      else:
        score = score_fn(x, torch.ones(data.shape[0], device=data.device) * t)
      x0t = (x + (1 - sde.alphas_cumprod[n]) * score) / sde.sqrt_alphas_cumprod[n]

      z = torch.randn_like(x)
      if i == sde.N - 1:
        z = torch.zeros_like(x)
      
      norm = torch.norm(Fy - measurement_model.A(x0t)) ** 2
      grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
      x0t_hat = x0t - reduced_alpha_cumprod[n] ** 2/ (2 * 0.001 ** 2) * grad / norm
      x.detach()

      eps_hat = (x - sde.sqrt_alphas_cumprod[n] * x0t_hat) / sde.sqrt_1m_alphas_cumprod[n]
      x = sde.sqrt_alphas_cumprod[n - 1] * x0t_hat + sde.sqrt_1m_alphas_cumprod[n - 1] * (torch.sqrt(1 - zeta) * eps_hat + torch.sqrt(zeta) * z)

      # x_0.requires_grad_()
      # norm = torch.norm(Fy - measurement_model.A(x_0)) ** 2
      # x_0_hat = x_0 - reduced_alpha_cumprod[n] / 2 * torch.autograd.grad(outputs=norm, inputs=x_0)[0] / measurement_model.norm_constant
      # epsilon_hat = (x - sqrt_alphas_cumprod[n] * x_0_hat) / sqrt_1m_alphas_cumprod[n]
      # x_0= sqrt_alphas_cumprod[n - 1] * x_0_hat + sqrt_1m_alphas_cumprod[n - 1] * (torch.sqrt(1 - zeta) * epsilon_hat + torch.sqrt(zeta) * torch.randn_like(x)) 


      if savedir is not None:
        if i % 100 == 0 or i == sde.N - 1:
          plt.imsave(savedir / f'score_{i}.png', clear(score), cmap='gray')
          plt.imsave(savedir / f'x0t_{i}.png', clear(x0t), cmap='gray')
          plt.imsave(savedir / f'x_{i}.png', clear(x), cmap='gray')
          plt.imsave(savedir / f'x0t_hat_{i}.png', clear(x0t_hat), cmap='gray')
          plt.imsave(savedir / f'eps_hat_{i}.png', clear(eps_hat), cmap='gray')

      # Collect values for visualization
      norms.append(norm.item())
      x0ts.append(torch.norm(x0t).item())
      x0t_hats.append(torch.norm(x0t_hat).item())
      scores.append(torch.norm(score).item())
      xs.append(torch.norm(x).item())
      eps_hats.append(torch.norm(eps_hat).item())
      etas.append((reduced_alpha_cumprod[n] ** 2/ (2 * 0.01 ** 2)).item())

    # Plot the collected values with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Norm', color='tab:blue')
    ax1.plot(norms, label='Norm', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Values', color='tab:orange')
    # ax2.plot(x0ts, label=r'$x_0^{(t)}$ Norm', color='tab:orange')
    # ax2.plot(x0t_hats, label=r'$\hat{x}_0^{(t)}$ Norm', color='tab:green')
    # ax2.plot(scores, label='Score Norm', color='tab:red')
    # ax2.plot(xs, label='x Norm', color='tab:purple')
    # ax2.plot(eps_hats, label=r'$\hat{\epsilon}$ Norm', color='tab:brown')
    ax2.plot(etas, label='1/eta', color='tab:pink')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    fig.legend(loc='upper right')
    plt.title('Values over Iterations')
    plt.savefig(savedir / 'values_over_iterations.png')
    plt.close()

    return inverse_scaler(x)

  return diffpir_sampler