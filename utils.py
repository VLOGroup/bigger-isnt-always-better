import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from fastmri_utils import fft2c_new, ifft2c_new
# from statistics import mean, stdev
from sigpy.mri import poisson, radial
from typing import Union
from skimage.morphology import skeletonize
from skimage.transform import rotate
from typing import Union, Tuple, Callable
import fastmri.models as fm

"""
Helper functions for new types of inverse problems
"""

def fft2(x):
  """ FFT with shifting DC to the center of the image"""
  return torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(x, dim=[-2, -1]), norm='ortho'), dim=[-2, -1])


def ifft2(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(x, dim=[-2, -1]), norm='ortho'), dim=[-2, -1])


def fft2_m(x):
  """ FFT for multi-coil """
  return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
  """ IFFT for multi-coil """
  return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))


def crop_center(img, cropx, cropy):
  c, y, x = img.shape
  startx = x // 2 - (cropx // 2)
  starty = y // 2 - (cropy // 2)
  return img[:, starty:starty + cropy, startx:startx + cropx]


def normalize(img):
  """ Normalize img in arbitrary range to [0, 1] """
  img -= torch.min(img)
  img /= torch.max(img)
  return img


def normalize_np(img):
  """ Normalize img in arbitrary range to [0, 1] """
  img -= np.min(img)
  img /= np.max(img)
  return img


def normalize_complex(img):
  """ normalizes the magnitude of complex-valued image to range [0, 1] """
  abs_img = normalize(torch.abs(img))
  ang_img = torch.angle(img)
  # original: ang_img = normalize(torch.angle(img))
  return abs_img * torch.exp(1j * ang_img)


class lambda_schedule:
  def __init__(self, total=2000):
    self.total = total

  def get_current_lambda(self, i):
    pass


class lambda_schedule_linear(lambda_schedule):
  def __init__(self, start_lamb=1.0, end_lamb=0.0):
    super().__init__()
    self.start_lamb = start_lamb
    self.end_lamb = end_lamb

  def get_current_lambda(self, i):
    return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / self.total)


class lambda_schedule_const(lambda_schedule):
  def __init__(self, lamb=1.0):
    super().__init__()
    self.lamb = lamb

  def get_current_lambda(self, i):
    return self.lamb


def clear(x):
  return x.detach().cpu().squeeze().numpy()

def clear_color(x):
  x = x.detach().cpu().squeeze().numpy()
  return np.transpose(x, (1, 2, 0))


def get_mask(img, size, batch_size, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False):
  mux_in = size ** 2
  if type.endswith('2d'):
    Nsamp = mux_in // acc_factor
  elif type.endswith('1d'):
    Nsamp = size // acc_factor
  if type == 'gaussian2d':
    mask = torch.zeros_like(img)
    cov_factor = size * (1.5 / 128)
    mean = [size // 2, size // 2]
    cov = [[size * cov_factor, 0], [0, size * cov_factor]]
    if fix:
      samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
      int_samples = samples.astype(int)
      int_samples = np.clip(int_samples, 0, size - 1)
      mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
    else:
      for i in range(batch_size):
        # sample different masks for batch
        samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
        int_samples = samples.astype(int)
        int_samples = np.clip(int_samples, 0, size - 1)
        mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
  elif type == 'uniformrandom2d':
    mask = torch.zeros_like(img)
    if fix:
      mask_vec = torch.zeros([1, size * size])
      samples = np.random.choice(size * size, int(Nsamp))
      mask_vec[:, samples] = 1
      mask_b = mask_vec.view(size, size)
      mask[:, ...] = mask_b
    else:
      for i in range(batch_size):
        # sample different masks for batch
        mask_vec = torch.zeros([1, size * size])
        samples = np.random.choice(size * size, int(Nsamp))
        mask_vec[:, samples] = 1
        mask_b = mask_vec.view(size, size)
        mask[i, ...] = mask_b
  elif type == 'gaussian1d':
    mask = torch.zeros_like(img)
    mean = size // 2
    std = size * (15.0 / 128)
    Nsamp_center = int(size * center_fraction)
    if fix:
      samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
      int_samples = samples.astype(int)
      int_samples = np.clip(int_samples, 0, size - 1)
      mask[... , int_samples] = 1
      c_from = size // 2 - Nsamp_center // 2
      mask[... , c_from:c_from + Nsamp_center] = 1
    else:
      for i in range(batch_size):
        samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp*1.2))
        int_samples = samples.astype(int)
        int_samples = np.clip(int_samples, 0, size - 1)
        mask[i, :, :, int_samples] = 1
        c_from = size // 2 - Nsamp_center // 2
        mask[i, :, :, c_from:c_from + Nsamp_center] = 1
  elif type == 'uniform1d':
    mask = torch.zeros_like(img)
    if fix:
      Nsamp_center = int(size * center_fraction)
      samples = np.random.choice(size, int(Nsamp - Nsamp_center))
      mask[..., samples] = 1
      # ACS region
      c_from = size // 2 - Nsamp_center // 2
      mask[..., c_from:c_from + Nsamp_center] = 1
    else:
      for i in range(batch_size):
        Nsamp_center = int(size * center_fraction)
        samples = np.random.choice(size, int(Nsamp - Nsamp_center))
        mask[i, :, :, samples] = 1
        # ACS region
        c_from = size // 2 - Nsamp_center // 2
        mask[i, :, :, c_from:c_from+Nsamp_center] = 1
  elif type == 'poisson':
    mask = poisson((size, size), accel=acc_factor, dtype=float)
    mask = torch.from_numpy(mask).cuda().float()
  else:
    NotImplementedError(f'Mask type {type} is currently not supported.')

  return mask


def get_radial_mask(
    shape,
    num_spokes,
    theta=np.pi * (3 - np.sqrt(5)),
    offset=0,
    theta0=0,
    skinny=True,
    extend=True
) -> torch.Tensor:
    if extend:
        mode = 'wrap'
    else:
        mode = 'constant'

    idx = np.zeros(shape, dtype=bool)
    idx0 = np.zeros(idx.shape, dtype=bool)
    idx0[int(shape[0] / 2), :] = 1

    for ii in range(num_spokes):
        idx1 = rotate(
            idx0,
            np.rad2deg(theta * (ii + offset) + theta0),
            resize=False,
            mode=mode
        ).astype(bool)
        if skinny:
            idx1 = skeletonize(idx1)
        idx |= idx1

    return torch.from_numpy(idx).cuda().float()[None, None]

def get_outer_mask(cutout):
    mask = np.ones((320, 320))

    x = np.repeat(np.arange(-(cutout // 2), (cutout // 2)), cutout + 1)
    y = np.tile(np.arange(-(cutout // 2), (cutout // 2)), cutout + 1)
    mask[160 + x, 160 + y] = 0
    return torch.from_numpy(mask.reshape((1, 1, 320, 320))).cuda().float()

def kspace_to_nchw(tensor):
    """
    Convert torch tensor in (Slice, Coil, Height, Width, Complex) 5D format to
    (N, C, H, W) 4D format for processing by 2D CNNs.

    Complex indicates (real, imag) as 2 channels, the complex data format for Pytorch.

    C is the coils interleaved with real and imaginary values as separate channels.
    C is therefore always 2 * Coil.

    Singlecoil data is assumed to be in the 5D format with Coil = 1

    Args:
        tensor (torch.Tensor): Input data in 5D kspace tensor format.
    Returns:
        tensor (torch.Tensor): tensor in 4D NCHW format to be fed into a CNN.
    """
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 5
    s = tensor.shape
    assert s[-1] == 2
    tensor = tensor.permute(dims=(0, 1, 4, 2, 3)).reshape(shape=(s[0], 2 * s[1], s[2], s[3]))
    return tensor


def nchw_to_kspace(tensor):
  """
  Convert a torch tensor in (N, C, H, W) format to the (Slice, Coil, Height, Width, Complex) format.

  This function assumes that the real and imaginary values of a coil are always adjacent to one another in C.
  If the coil dimension is not divisible by 2, the function assumes that the input data is 'real' data,
  and thus pads the imaginary dimension as 0.
  """
  assert isinstance(tensor, torch.Tensor)
  assert tensor.dim() == 4
  s = tensor.shape
  if tensor.shape[1] == 1:
    imag_tensor = torch.zeros(s, device=tensor.device)
    tensor = torch.cat((tensor, imag_tensor), dim=1)
    s = tensor.shape
  tensor = tensor.view(size=(s[0], s[1] // 2, 2, s[2], s[3])).permute(dims=(0, 1, 3, 4, 2))
  return tensor


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim)).unsqueeze(dim)


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def restore_checkpoint(ckpt_dir, state, device, skip_sigma=False):
  loaded_state = torch.load(ckpt_dir, map_location=device)
  loaded_model_state = loaded_state['model']
  if skip_sigma:
    loaded_model_state.pop('module.sigmas')

  state['model'].load_state_dict(loaded_model_state, strict=False)
  state['ema'].load_state_dict(loaded_state['ema'])
  state['step'] = loaded_state['step']
  print(f'loaded checkpoint dir from {ckpt_dir}')
  return state

def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)


#####################################################
# from Zach et al.
#####################################################
  
def psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    value_range: Union[torch.Tensor, float] = 1.
) -> torch.Tensor:
    return (10 * torch.log10(value_range**2 / mse(x, y)))


def nmse(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return torch.sum((x - y)**2, dim=(1, 2, 3),
                  keepdim=True) / torch.sum(x**2, dim=(1, 2, 3), keepdim=True)


def mse(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return torch.mean((x - y)**2, dim=(1, 2, 3), keepdim=True)


class SSIM(torch.nn.Module):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer(
            "w",
            torch.ones(1, 1, win_size, win_size) / win_size**2
        )
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)
        self.w: torch.Tensor

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: Union[torch.Tensor, float] = 1.,
        reduced: bool = True,
    ):
        C1 = (self.k1 * data_range)**2
        C2 = (self.k2 * data_range)**2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced:
            return S.mean((1, 2, 3))
        else:
            return S

class AsMatrix():
    def __init__(
        self,
        operator,
        adjoint,
    ):
        self.operator = operator
        self.adjoint = adjoint

    def __matmul__(
        self,
        x,
    ):
        return self.operator(x)

    def forward(self, x):
        return self @ x

    def __call__(self, x):
        return self @ x

    @property
    def H(self, ):
        return AsMatrix(self.adjoint, self.operator)

def inner(x, y, keepdim=False):
    if torch.is_complex(x) or torch.is_complex(y):
        prod = (
            torch.view_as_real(x.to(torch.complex64)) *
            torch.view_as_real(y.to(torch.complex64))
        ).sum((1, 2, 3, 4))
        return prod[:, None, None, None] if keepdim else prod
    else:
        return (x * y).sum((1, 2, 3), keepdim=keepdim)
    
class Div(nn.Module):
    def __init__(self):
        super().__init__()

    def __matmul__(self, x):
        div = x.new_zeros(x.shape[:-1])
        div[:, :, :, 1:] += x[:, :, :, :-1, 0]
        div[:, :, :, :-1] -= x[:, :, :, :-1, 0]
        div[:, :, 1:, :] += x[:, :, :-1, :, 1]
        div[:, :, :-1, :] -= x[:, :, :-1, :, 1]
        return div

    def forward(self, x):
        return self @ x
    
class Grad(nn.Module):
    def __init__(self):
        super().__init__()

    def __matmul__(self, x):
        grad = x.new_zeros((*x.shape, 2))
        grad[:, :, :, :-1, 0] += x[:, :, :, 1:] - x[:, :, :, :-1]
        grad[:, :, :-1, :, 1] += x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad

    def forward(self, x):
        return self @ x

def apgd(
    x_init: torch.Tensor,
    f_nabla: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    f: Callable[[torch.Tensor], torch.Tensor],
    prox: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    callback: Callable[[torch.Tensor, int], None] = lambda x, i: None,
    max_iter: int = 200,
    gamma: float = 1.,
):
    x = x_init.clone()
    x_old = x.clone()
    L = 1 * torch.ones((x.shape[0], 1, 1, 1), dtype=torch.float32, device=x.device)
    for i in range(max_iter):
        beta = i / (i + 3) * 1
        x_bar = x + beta * (x - x_old)
        x_old = x.clone()
        n = torch.randn_like(x) * 7.5e-3 * gamma
        energy, grad = f_nabla(x_bar + n)
        for _ in range(10):
            x = prox(x_bar - grad / L, 1 / L)
            dx = x - x_bar
            bound = energy + inner(grad, dx, keepdim=True) \
                + L * inner(dx, dx, keepdim=True) / 2
            if torch.all((energy_new := f(x + n)) <= bound):
                break
            L = torch.where(energy_new <= bound, L, 2 * L)
        L /= 1.5
        callback(x, i)
    return x

class CharbTV(nn.Module):
    def __init__(
        self,
        eps: float = 1e-2,
    ):
        super().__init__()
        self.nabla = Grad()
        self.div = Div()
        self.eps = eps

    def forward(self, x):
        return self.energy(x)

    def energy(self, x):
        nabla_u = self.nabla @ x
        norm_Du = torch.sqrt((nabla_u**2).sum(dim=-1) + self.eps**2)
        e = norm_Du.sum((1, 2, 3), keepdim=True)
        return e

    def grad(self, x):
        nabla_u = self.nabla @ x
        norm_Du = torch.sqrt((nabla_u**2).sum(dim=-1) + self.eps**2)
        e = norm_Du.sum((1, 2, 3), keepdim=True)
        g = self.div @ (nabla_u / norm_Du[..., None])
        return e, g

def unet_model(path):
    R = fm.Unet(
        in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0
    )
    state_dict = torch.load(path)
    R.load_state_dict(state_dict)
    R = R.eval()
    return R.cuda()

def unet_normalize(
    data: torch.Tensor,
    eps: Union[float, torch.Tensor] = 1e-11
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = data.mean(dim=(1, 2, 3), keepdim=True)
    std = data.std(dim=(1, 2, 3), keepdim=True)
    normalized = (data - mean) / (std + eps)
    return normalized, mean, std

def call_unet(net, samples):
    normalized, mean, std = unet_normalize(samples)
    out = torch.stack([net(im[None])[0] for im in normalized])
    return out * std + mean

def batchfy(tensor, batch_size):
  n = len(tensor)
  num_batches = n // batch_size + 1
  return tensor.chunk(num_batches, dim=0)