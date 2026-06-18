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
import argparse
import yaml
from tqdm import trange

def evaluate(model_config, num_iterations, savedir, device, logger=None):
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

  logger.info(f'Model Architecture:\n{score_model.all_modules}')
  
  try:
    score_model.eval()
    accumulated_grad = torch.zeros((320, 320)).to(device)
    
    print(f"Averaging ERF over {num_iterations} iterations...")

    for i in trange(num_iterations):
        # 1. Create a random input (Gaussian noise)
        input_image = torch.randn(1, 1, 320, 320, requires_grad=True, device=device)
        
        # 2. Forward pass (extracting layer4 specifically for ResNet)
        # Using the same path as before, but slightly more condensed
        x = score_model(input_image, 10 * torch.rand(1, device=device))
        
        # 3. Isolate central unit
        _, c, h, w = x.shape
        central_unit = x[0, 0, h//2, w//2]
        # print(central_unit)
        
        # 4. Backward pass
        score_model.zero_grad()
        central_unit.backward()
        # print(input_image.grad)
        
        # 5. Accumulate the absolute gradient magnitude
        # We take the absolute value so that negative and positive influence don't cancel out
        grad = input_image.grad.detach()[0]
        grad_abs = torch.abs(grad).squeeze() # Average over channels
        accumulated_grad += grad_abs
        
    # 6. Final processing
    # Average and normalize to [0, 1]
    erf_map = accumulated_grad / num_iterations
    erf_map = (erf_map - erf_map.min()) / (erf_map.max() - erf_map.min())
    erf_map = erf_map.cpu().numpy()
    
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(erf_map, cmap='viridis', vmin=0, vmax=0.1)
    plt.title("Linear Scale ERF")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(erf_map + 1e-7), cmap='viridis')
    plt.title("Log Scale ERF (Shows boundaries better)")
    plt.colorbar()

    plt.show()



    # 1. Sort all pixel values in descending order
    flat_erf = erf_map.flatten()
    sorted_erf = np.sort(flat_erf)[::-1]
    
    # 2. Find the cumulative sum and the cutoff value for the threshold
    cum_sum = np.cumsum(sorted_erf)
    total_sum = cum_sum[-1]
    cutoff_idx = np.where(cum_sum >= total_sum * 0.8)[0][0]
    cutoff_value = sorted_erf[cutoff_idx]
    
    # 3. Create a mask of pixels that contribute to the top X%
    mask = erf_map >= cutoff_value
    
    # 4. Find the bounding box of these pixels
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    height = y_max - y_min
    width = x_max - x_min

    print(f"ERF 90% Bounding Box: {height}x{width} pixels")
    print(f"Effective Diameter (Avg): {(height + width) / 2:.2f} pixels")

    # Visualize the 'Active' Zone
    plt.imshow(mask, cmap='gray')
    plt.title(f"90% Impact Zone ({height}x{width} pixels)")
    plt.show()

  except Exception as e:
    logger.error("An error occurred during evaluation: %s", e)
    raise

  return


def create_argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_config', type=str, help='which model config file to use', required=True)
  parser.add_argument('--num_iter', type=int, help='number of iterations for ERF averaging', required=True)
  parser.add_argument('--savedir', type=str, help='where to store the results', required=True)
  return parser

def load_config(config_file):
  with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
  return config

def main():
  try:
    args = create_argparser().parse_args()
    model_config = load_config(args.model_config)
    num_iterations = args.num_iter
    # Create the working directory
    savedir = Path(args.savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    logger = logging.getLogger(name='training')
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(savedir / 'training.log')
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.info(f'Evaluating {args.model_config} with {args.num_iter} iterations for ERF averaging')
    # Run the training pipeline

    # check if cuda is available:
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    evaluate(model_config, num_iterations, savedir, device=device, logger=logger)

    # save_config(model_config, args.model_config)
  except Exception as e:
    logging.error("An error occurred: %s", e)
    raise


if __name__ == "__main__":
  main()
