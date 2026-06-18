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

"""Training and evaluation"""

import logging
from pathlib import Path
from evaluation_script import evaluate
import argparse
import yaml
import torch

def create_argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_config', type=str, help='which model config file to use', required=True)
  parser.add_argument('--data_config', type=str, help='which data config file to use', required=True)
  parser.add_argument('--evaluation_config', type=str, help='which training config file to use', required=True)
  parser.add_argument('--sample_config', type=str, help='which sample config file to use', required=True)
  parser.add_argument('--savedir', type=str, help='where to store the results', required=True)
  return parser

def load_config(config_file):
  with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
  return config

def save_config(config, config_file):
  with open(config_file, 'w') as file:
    yaml.dump(config, file)


def main():
  try:
    args = create_argparser().parse_args()
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)
    evaluation_config = load_config(args.evaluation_config)
    sample_config = load_config(args.sample_config)
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
    logger.info(f'Evaluating {args.model_config} on {args.data_config} dataset with {args.evaluation_config} and sampling via {args.sample_config}')
    # Run the training pipeline

    # check if cuda is available:
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    evaluate(model_config, data_config, evaluation_config, sample_config, savedir, device=device, logger=logger)

    # save_config(model_config, args.model_config)
  except Exception as e:
    logging.error("An error occurred: %s", e)
    raise


if __name__ == "__main__":
  main()
