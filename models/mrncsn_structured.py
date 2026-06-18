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

from . import utils, layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='mrncsn_structured')
class MRNCSN_STRUCTURED(nn.Module):
  """MRNCSN model"""

  def __init__(self, config):
    super().__init__()
    print('MRNCSN_STRUCTURED model')
    self.config = config
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config['model']['nf']
    self.num_res_blocks = num_res_blocks = config['model']['num_res_blocks']
    self.attn_resolutions = attn_resolutions = config['model']['attn_resolutions']
    dropout = config['model']['dropout']
    resamp_with_conv = config['model']['resamp_with_conv']
    self.num_resolutions = num_resolutions = config['model']['num_resolutions']

    self.conditional = conditional = config['model']['conditional']  # noise-conditional
    fir = config['model']['fir']
    fir_kernel = config['model']['fir_kernel']
    self.skip_rescale = skip_rescale = config['model']['skip_rescale']
    self.resblock_type = resblock_type = config['model']['resblock_type'].lower()
    self.progressive = progressive = config['model']['progressive'].lower()
    self.progressive_input = progressive_input = config['model']['progressive_input'].lower()
    self.embedding_type = embedding_type = config['model']['embedding_type'].lower()
    init_scale = config['model']['init_scale']
    assert progressive in ['none', 'output_skip', 'residual']
    assert progressive_input in ['none', 'input_skip', 'residual']
    assert embedding_type in ['fourier', 'positional']
    combine_method = config['model']['progressive_combine'].lower()
    combiner = functools.partial(Combine, method=combine_method)

    # modules will be a list of nn.ModuleLists
    modules = [  [] for _ in range(num_resolutions) ]
    # timestep/noise_level embedding; only for continuous training
    if embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      # assert config.training.continuous, "Fourier features are only used for continuous training."

      modules[0].append(layerspp.GaussianFourierProjection(
        embedding_size=nf, scale=config['model']['fourier_scale']
      ))
      embed_dim = 2 * nf

    elif embedding_type == 'positional':
      embed_dim = nf

    else:
      raise ValueError(f'embedding type {embedding_type} unknown.')

    if conditional:
      modules[0].append(nn.Linear(embed_dim, nf * 4))
      modules[0][-1].weight.data = default_initializer()(modules[0][-1].weight.shape)
      nn.init.zeros_(modules[0][-1].bias)
      modules[0].append(nn.Linear(nf * 4, nf * 4))
      modules[0][-1].weight.data = default_initializer()(modules[0][-1].weight.shape)
      nn.init.zeros_(modules[0][-1].bias)

    AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive == 'output_skip':
      self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = functools.partial(layerspp.Upsample,
                                           fir=fir, fir_kernel=fir_kernel, with_conv=True)

    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive_input == 'input_skip':
      self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = functools.partial(layerspp.Downsample,
                                             fir=fir, fir_kernel=fir_kernel, with_conv=True)

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block

    channels = 1
    if progressive_input != 'none':
      input_pyramid_ch = channels

    # Initialize different input layers
    modules[0].append(conv3x3(channels, nf))
    # input_modules = [conv3x3(channels, nf) for i in range(num_resolutions)]
    hs_c = [nf]

    in_ch = nf
    # input_pyramid_ch = in_ch
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf  # * ch_mult[i_level]
        modules[i_level].append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules[i_level+1].append(Downsample(in_ch=in_ch))
        else:
          modules[i_level+1].append(ResnetBlock(down=True, in_ch=in_ch))

        if progressive_input == 'input_skip':
          modules[i_level+1].append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
          if combine_method == 'cat':
            in_ch *= 2

        elif progressive_input == 'residual':
          modules[i_level+1].append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
          input_pyramid_ch = in_ch

        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules[-1].append(ResnetBlock(in_ch=in_ch))
    modules[-1].append(ResnetBlock(in_ch=in_ch))

    pyramid_ch = 0
    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf  # * ch_mult[i_level]
        modules[i_level].append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                   out_ch=out_ch))
        in_ch = out_ch

      if progressive != 'none':
        if i_level == num_resolutions - 1:
          if progressive == 'output_skip':
            modules[i_level].append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules[i_level].append(conv3x3(in_ch, channels, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules[i_level].append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules[i_level].append(conv3x3(in_ch, in_ch, bias=True))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name.')
        else:
          if progressive == 'output_skip':
            modules[i_level].append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules[i_level].append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules[i_level].append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name')

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules[i_level].append(Upsample(in_ch=in_ch))
        else:
          modules[i_level].append(ResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c

    if progressive != 'output_skip':
      modules[0].append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
      # Initialize all output layers
      modules[0].append(conv3x3(in_ch, channels, init_scale=init_scale))
      # output_modules = [conv3x3(in_ch, channels, init_scale=init_scale) for i in range(num_resolutions)]

    modules = [nn.ModuleList(m) for m in modules]
    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, time_cond, d):
    # timestep/noise_level embedding; only for continuous training
    modules = self.all_modules
    # input_modules = self.input_modules
    # output_modules = self.output_modules
    m_idx = [0 for _ in range(self.num_resolutions)]
    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      used_sigmas = time_cond
      temb = modules[0][m_idx[0]](torch.log(used_sigmas))
      m_idx[0] += 1

    elif self.embedding_type == 'positional':
      # Sinusoidal positional embeddings.
      timesteps = time_cond
      used_sigmas = self.sigmas[time_cond.long()]
      temb = layers.get_timestep_embedding(timesteps, self.nf)

    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    if self.conditional:
      temb = modules[0][m_idx[0]](temb)
      m_idx[0] += 1
      temb = modules[0][m_idx[0]](self.act(temb))
      m_idx[0] += 1
    else:
      temb = None

    if not self.config['model']['data_centered']:
      # If input data is in [0, 1]
      x = 2 * x - 1.

    # Downsampling block
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x

    # Choose corresponding input layer
    hs = [modules[0][m_idx[0]](x)]
    m_idx[0] += 1
    # hs = [input_modules[d - 1](x)]

    for i_level in range(d):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[i_level][m_idx[i_level]](hs[-1], temb)
        m_idx[i_level] += 1

        hs.append(h)

      if i_level != d - 1:
        if self.resblock_type == 'ddpm':
          h = modules[i_level+1][m_idx[i_level+1]](hs[-1])
          m_idx[i_level+1] += 1
        else:
          h = modules[i_level+1][m_idx[i_level+1]](hs[-1], temb)
          m_idx[i_level+1] += 1

        # debug
        if self.progressive_input == 'input_skip':
          input_pyramid = self.pyramid_downsample(input_pyramid)
          h = modules[i_level+1][m_idx[i_level+1]](input_pyramid, h)
          m_idx[i_level+1] += 1

        elif self.progressive_input == 'residual':
          input_pyramid = modules[i_level+1][m_idx[i_level+1]](input_pyramid)
          m_idx[i_level+1] += 1
          if self.skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid
          # else:
          #   if d != self.num_resolutions:
          #     m_idx[i_level] += 2

        hs.append(h)

    h = hs[-1]
    if d == self.num_resolutions:
      h = modules[-1][m_idx[-1]](h, temb)
      m_idx[-1] += 1
      h = modules[-1][m_idx[-1]](h, temb)
      m_idx[-1] += 1
    # else:
    #   m_idx =  51 - 6*d # O: fix this

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(d)):
      for i_block in range(self.num_res_blocks + 1):
        tmp = hs.pop()
        h = modules[i_level][m_idx[i_level]](torch.cat([h, tmp], dim=1), temb)
        m_idx[i_level] += 1


      if self.progressive != 'none':
        if i_level == self.num_resolutions - 1:
          if self.progressive == 'output_skip':
            pyramid = self.act(modules[i_level][m_idx[i_level]](h))
            m_idx[i_level] += 1
            pyramid = modules[i_level][m_idx[i_level]](pyramid)
            m_idx[i_level] += 1
          elif self.progressive == 'residual':
            pyramid = self.act(modules[i_level][m_idx[i_level]](h))
            m_idx[i_level] += 1
            pyramid = modules[i_level][m_idx[i_level]](pyramid)
            m_idx[i_level] += 1
          else:
            raise ValueError(f'{self.progressive} is not a valid name.')
        else:
          if self.progressive == 'output_skip':
            pyramid = self.pyramid_upsample(pyramid)
            pyramid_h = self.act(modules[i_level][m_idx[i_level]](h))
            m_idx[i_level] += 1
            pyramid_h = modules[i_level][m_idx[i_level]](pyramid_h)
            m_idx[i_level] += 1
            pyramid = pyramid + pyramid_h
          elif self.progressive == 'residual':
            pyramid = modules[i_level][m_idx[i_level]](pyramid)
            m_idx[i_level] += 1
            if self.skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{self.progressive} is not a valid name')

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[i_level][m_idx[i_level]](h)
          m_idx[i_level] += 1
        else:
          h = modules[i_level][m_idx[i_level]](h, temb)
          m_idx[i_level] += 1

    assert not hs

    if self.progressive == 'output_skip':
      h = pyramid
    else:
      h = self.act(modules[0][m_idx[0]](h))
      m_idx[0] += 1
      # Choose corresponding output layer
      h = modules[0][m_idx[0]](h)
      m_idx[0] += 1
      # h = output_modules[d - 1](h)

    for i in range(d):
      assert m_idx[i] == len(modules[i])
    if self.config['model']['scale_by_sigma']:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas
    return h
