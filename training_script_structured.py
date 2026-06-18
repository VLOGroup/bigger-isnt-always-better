import time
import numpy as np
import logging
from models import mrncsn, msncsn, ncsnpp, mrncsn_structured
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import mydata
import sde_lib
import torch
from torch.utils import tensorboard
from utils import save_checkpoint, restore_checkpoint
import imageio.v3 as iio
from pathlib import Path

def train(model_config, data_config, training_config, sample_config, workdir, device, logger=None):
  if logger is None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

  # Create directories for experimental logs
  sample_dir = workdir / 'samples'
  sample_dir.mkdir(parents=True, exist_ok=True)

  tb_dir = workdir  / 'tensorboard'
  tb_dir.mkdir(parents=True, exist_ok=True)

  # Initialize model.
  score_model = mutils.create_model(model_config)
  score_model = score_model.to(device)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=model_config['model']['ema_rate'])
  optimizer = losses.get_optimizer(training_config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  logger.info(f'Model Architecture:\n{score_model.all_modules}')

  checkpoint_dir = workdir / 'checkpoints'
  checkpoint_dir.mkdir(parents=True, exist_ok=True)

  if (checkpoint_dir / 'checkpoint.pth').is_file():
    state = restore_checkpoint(checkpoint_dir / 'checkpoint.pth', state, device)
  # score_model.all_modules[1].requires_grad_(False)

  initial_step = int(state['step'])
  writer = tensorboard.SummaryWriter(tb_dir, purge_step=initial_step)

  # Build pytorch dataloader for training
  if data_config['dataset'] == 'fastmri_knee':
    train_dl, _ = mydata.create_dataloader(Path(data_config['root']))
  elif data_config['dataset'] == 'celeba':
    train_dl = mydata.create_celeba_dataloader(Path(data_config['root']), data_config['image_size'])
  elif data_config['dataset'] == 'ct':
    train_dl, _ = mydata.create_ct_dataloader(Path(data_config['root']), data_config['image_size'], batch_size=training_config['training']['batch_size'])
  num_data = len(train_dl.dataset)
  loss_over_iter = np.zeros(training_config['training']['epochs'] * num_data)
  if (workdir / 'loss.npy').is_file():
    loss_over_iter_temp = np.load(workdir / 'loss.npy')
    loss_over_iter[:len(loss_over_iter_temp)] = loss_over_iter_temp

  # Get number of previously trained epochs
  initial_batch = initial_step // num_data

  # Create data normalizer and its inverse
  scaler = mydata.get_data_scaler(data_config)
  inverse_scaler = mydata.get_data_inverse_scaler(data_config)

  # Setup SDEs
  if model_config['sde']['sde'].lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=model_config['model']['beta_min'], beta_max=model_config['model']['beta_max'], N=model_config['model']['num_scales'])
    sampling_eps = 1e-3
  elif model_config['sde']['sde'].lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=model_config['model']['beta_min'], beta_max=model_config['model']['beta_max'], N=model_config['model']['num_scales'])
    sampling_eps = 1e-3
  elif model_config['sde']['sde'].lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=model_config['model']['sigma_min'], sigma_max=model_config['model']['sigma_max'], N=model_config['model']['num_scales'])
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {model_config['sde']['sde']} unknown.")
  
  

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(training_config)
  continuous = model_config['sde']['continuous']
  reduce_mean = model_config['sde']['reduce_mean']
  likelihood_weighting = model_config['sde']['likelihood_weighting']
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting, depth_param=True)

  # Building sampling functions
  if training_config['training']['snapshot_sampling']:
    sampling_shape = (1 , data_config['num_channels'], data_config['image_size'], data_config['image_size'])
    sampling_fn = sampling.get_sampling_fn(sample_config, sde, sampling_shape, inverse_scaler, continuous, sampling_eps, device, depth_param=True)

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logger.info(f'Starting training loop at step {initial_step} / batch {initial_batch}.')
  training_times_batch = np.zeros(training_config['training']['epochs'])
  training_times_epoch = np.zeros(training_config['training']['epochs'])
  if (workdir / 'training_times_epoch.csv').is_file():
    training_times_epoch_temp = np.loadtxt(workdir / 'training_times_epoch.csv', dtype=float, delimiter=',')
    training_times_epoch[:len(training_times_epoch_temp)] = training_times_epoch_temp
  if (workdir / 'training_times_batch.csv').is_file():
    training_times_batch_temp = np.loadtxt(workdir / 'training_times_batch.csv', dtype=float, delimiter=',')
    training_times_batch[:len(training_times_batch_temp)] = training_times_batch_temp

  # adopt ema
  for p in state['model'].all_modules.parameters():
    p.requires_grad = True
  ema = ExponentialMovingAverage(state['model'].parameters(), decay=model_config['model']['ema_rate'])
  state['ema'] = ema

  for epoch in range(initial_batch, training_config['training']['epochs']):
    # d = epoch//(training_config['training']['epochs'] // model_config['model']['num_resolutions']) + 1
    # for p in state['model'].all_modules.parameters():
    #   p.requires_grad = True
    # state['optimizer'].zero_grad()
    # for p in state['model'].all_modules.parameters():
    #   p.requires_grad = False
    # for p in state['model'].all_modules[d-1].parameters():
    #   p.requires_grad = True
    # if d != (epoch - 1)//((training_config['training']['epochs']) // model_config['model']['num_resolutions']) + 1 or epoch == 0:
    #   ema = ExponentialMovingAverage(state['model'].parameters(), decay=model_config['model']['ema_rate'])
    #   state['ema'] = ema
    logger.info('=================================================')
    logger.info(f'Epoch: {epoch}/{training_config["training"]["epochs"]}')
    logger.info('=================================================')
    tic_epoch = time.time()

    for step, batch in enumerate(train_dl):
      try:
        tic_batch = time.time()
        batch = scaler(batch.to(device)).view(-1, 1, 320, 320)
        loss = train_step_fn(state, batch)
        loss_over_iter[epoch * num_data + step] = loss
        training_times_batch[epoch] += time.time() - tic_batch
        if step % training_config['training']['log_freq'] == 0:
          logger.info("step: %d, training_loss: %.5e" % (step, loss.item()))
          global_step = num_data * epoch + step
          writer.add_scalar("training_loss", scalar_value=loss, global_step=global_step)
          logger.info(f'CUDA  Memory Allocated: {torch.cuda.memory_allocated()}')

      except Exception as e:
        logger.error("An error occurred during training: %s", e)
        raise
    
    state['optimizer'].zero_grad()

    training_times_epoch[epoch] = time.time() - tic_epoch
    np.savetxt(workdir / 'training_times_epoch.csv', training_times_epoch, delimiter=',')
    training_times_batch[epoch] /= num_data
    np.savetxt(workdir / 'training_times_batch.csv', training_times_batch, delimiter=',')

    np.save(workdir / 'loss', loss_over_iter)

    # save checkpoint
    save_checkpoint(checkpoint_dir / 'checkpoint.pth', state)

    # Generate and save samples for every epoch
    if training_config['training']['snapshot_sampling'] and (epoch % training_config['training']['sample_frequency'] == 0 or epoch == training_config['training']['epochs'] - 1):
      ema.store(score_model.parameters())
      ema.copy_to(score_model.parameters())

      this_sample_dir = sample_dir  / f'iter_{epoch}'
      this_sample_dir.mkdir(parents=True, exist_ok=True)
      try:
        with torch.no_grad():
          for d in range(1, model_config['model']['num_resolutions'] + 1):
            sample, _ = sampling_fn(score_model, d=d)
            sample_to_save = sample.squeeze().detach().cpu().numpy()
            np.save(this_sample_dir / f'sample_d{d}.np', sample_to_save)

            iio.imwrite(this_sample_dir /  f'sample_d{d}.png', np.clip(sample_to_save * 255, 0, 255).astype(np.uint8))
            del sample_to_save
            del sample
      except Exception as e:
        logger.error("An error occurred during sampling: %s", e)
        raise
      ema.restore(score_model.parameters())
      torch.cuda.empty_cache()
      logger.info(f'CUDA  Memory Allocated: {torch.cuda.memory_allocated()}')
  return
