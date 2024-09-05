from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple

import h5py
import pandas as pd
import torch as th
import pickle
import random
import matplotlib.pyplot as plt

from utils import fft2, ifft2, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint, normalize_complex, ifft2_m, fft2_m, normalize
from torchvision.transforms.functional import center_crop
from torchvision.transforms import v2
from PIL import Image
import glob
import ast

def select_slices(filename, number_of_slices):
    if number_of_slices % 2 == 0:
        return [(filename, slice) for slice in range(-(number_of_slices // 2), number_of_slices // 2)]
    return [(filename, slice) for slice in range(-(number_of_slices // 2), number_of_slices // 2 + 1)]

def select_brain_slices(filename, number_of_slices):
    return [(filename, slice) for slice in range(number_of_slices)]

class MRIDataset(DataLoader):
    def __init__(self, root, split='train', is_complex=False):
        self.root = root
        self.is_complex = is_complex

        self.filenames = open(f'./split/corpd/{split}.txt', 'r').read().splitlines()
        self.data = []
        for filename in self.filenames:
            self.data.extend(select_slices(filename, 15))

    def __getitem__(self, idx):
        filename = self.data[idx][0]
        slice = self.data[idx][1]
        file = h5py.File(self.root / filename, 'r')
        data = file['reconstruction_rss'][len(file['reconstruction_rss']) // 2 + slice]
        data = data - np.min(data)
        return data / np.max(data)

    def __len__(self):
        return len(self.data)
    
class MRITestDataset(DataLoader):
    def __init__(self, root, split='test', fat_suppression=False, is_complex=True, PI=False):
        self.root = root
        self.is_complex = is_complex
        self.PI = PI
        if split == 'test' and fat_suppression == False:
            self.filenames = open(f'./split/corpd/validation.txt', 'r').read().splitlines()
            self.filenames += open(f'./split/corpd/test.txt', 'r').read().splitlines()
        elif split == 'test' and fat_suppression == True:
            self.filenames = open(f'./split/corpdfs/test.txt', 'r').read().splitlines()
        random.seed(42)
        self.filenames = random.sample(self.filenames, 50)
        open(f'./split/{"corpdfs" if fat_suppression else "corpd"}/test_selected.txt', 'w').write('\n'.join(self.filenames))
        self.data = []
        for filename in self.filenames:
            self.data.extend(select_slices(filename, 10))

    def __getitem__(self, idx):
        filename = self.data[idx][0]
        slice = self.data[idx][1]
        file = h5py.File(self.root / filename, 'r')
        if self.PI:
            data = normalize_complex(center_crop(ifft2_m(th.from_numpy(file['kspace'][len(file['reconstruction_rss']) // 2 + slice])), (320, 320)))
        else:
            data = normalize(th.from_numpy(file['reconstruction_rss'][len(file['reconstruction_rss']) // 2 + slice]))
        return data

    def __len__(self):
        return len(self.data)
    
class MRITestBrainDataset(DataLoader):
    def __init__(self, root, split='test', is_complex=True, PI=False):
        self.root = root
        self.is_complex = is_complex
        self.filenames = open(f'./split/brain/test_selected.txt', 'r').read().splitlines()
        self.data = []
        self.PI = PI
        for filename in self.filenames:
            self.data.extend(select_brain_slices(filename, 5))


    def __getitem__(self, idx):
        filename = self.data[idx][0]
        slice = self.data[idx][1]
        file = h5py.File(self.root / filename, 'r')
        if self.PI:
            data = center_crop(ifft2_m(th.from_numpy(file['kspace'][slice])), (320, 320))
            print(data.shape)
            print(th.sum(data, axis=0).shape)
            print(center_crop(th.from_numpy(file['reconstruction_rss'][slice]), (320, 320)).shape)
            print(th.sum(th.sum(data, axis=0) - center_crop(th.from_numpy(file['reconstruction_rss'][slice]), (320, 320))))
        else:
            data = normalize(center_crop(th.from_numpy(file['reconstruction_rss'][slice]), (320, 320)))
        return data

    def __len__(self):
        return len(self.data)
    

class MRITestProstateDataset(DataLoader):
    def __init__(self, root, split='test', is_complex=True):
        self.root = root
        self.is_complex = is_complex
        self.filenames = os.listdir(root)
        #random.seed(42)
        #self.filenames = random.sample(self.filenames, 100)
        #open(f'./split/brain/test_selected.txt', 'w').write('\n'.join(self.filenames))
        self.data = []
        for filename in self.filenames[:-1]:
            self.data.extend(select_slices(filename, 11))
        self.data.extend(select_slices(self.filenames[-1], 5))


    def __getitem__(self, idx):
        filename = self.data[idx][0]
        slice = self.data[idx][1]
        file = h5py.File(self.root / filename, 'r')
        data = normalize(th.from_numpy(file['reconstruction_rss'][slice]))
        return data

    def __len__(self):
        return len(self.data)


class MRIDataset_infer(DataLoader):
    def __init__(self, root, sort=True, split='train', is_complex=False):
        self.root = root
        self.is_complex = is_complex

        self.filenames = open(f'./split/corpd/{split}.txt', 'r').read().splitlines()
        self.data = [select_slices(filename, 15) for filename in self.filenames]

    def __getitem__(self, idx):
        filename = self.data[idx][0]
        slice = self.data[idx][1]
        file = h5py.File(self.root / filename, 'r')
        data = file['reconstruction_rss'][len(file['reconstruction_rss']) // 2 + slice]
        data = data - np.min(data)
        return data / np.max(data), filename

    def __len__(self):
        return len(self.data)


def create_dataloader(evaluation=False, sort=True, fat_suppression=False, PI=False):
    shuffle = True if not evaluation else False
    train_dataset = MRIDataset(Path('/srv/local/lg/FAST_MRI') / f'singlecoil_train')
    val_dataset = MRITestDataset(Path('/srv/local/lg/FAST_MRI') / f'multicoil_val', fat_suppression=fat_suppression, PI=PI)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1, #configs.training.batch_size,
        shuffle=shuffle,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1, #configs.training.batch_size,
        shuffle=False,
        # shuffle=True,
        drop_last=True
    )
    return train_loader, val_loader

def create_brain_dataloader(PI=False):
    val_dataset = MRITestBrainDataset(Path('/srv/local/lg/Brain_MRI') / f'multicoil_test_full', PI=PI)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True
    )
    return val_loader

def create_prostate_dataloader():
    val_dataset = MRITestProstateDataset(Path('/data/glaszner/prostate') / f'test_T2_1')
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True
    )
    return val_loader


class CelebADataset(DataLoader):
    def __init__(self, root, size):
        self.root = root
        self.data = [str(i) for i in range(30_000)]

        if os.path.exists(root + '/resized'):
            return

        os.mkdir(root + '/resized')
        transform = v2.Compose([
            v2.PILToTensor(),
            v2.Grayscale(),
            v2.Resize((320, 320), antialias=True)
            ])
        for filename in self.data:
            img = Image.open(self.root + '/' + filename + '.jpg')
            img = transform(img)
            img = img - th.min(img)
            img = img / th.max(img)
            np.save(root + '/resized/' + filename, img.numpy())


    def __getitem__(self, idx):
        filename = self.data[idx]
        data = np.load(self.root + '/resized/' + filename + '.npy')
        return data 

    def __len__(self):
        return len(self.data)

def create_celeba_dataloader(size):
    train_dataset = CelebADataset('/srv/local/lg/CelebA-HQ-img', size)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True
    )
    return train_loader


class CTDataset(DataLoader):
    def __init__(self, root, size):
        self.root = root
        self.filenames = glob.glob(root + '*.hdf5')
        self.data = []

        if not os.path.exists(root + '/resized'):
            os.mkdir(root + '/resized')
            transform = v2.Compose([
                v2.RandomRotation([-90, -90]),
                v2.Resize((320, 320), antialias=True)
                ])
            for filename in self.filenames:
                img = transform(th.unsqueeze(th.from_numpy(h5py.File(filename, 'r')['data'][:, :, :]), 1))
                img = img - th.min(img)
                img = img / th.max(img)
                np.save(root + '/resized/' + filename[-27:-5], img.numpy())

        for filename in self.filenames[:-30]:
            self.data.extend(select_slices(filename, 120))


    def __getitem__(self, idx):
        filename = self.data[idx][0]
        slice = self.data[idx][1]
        data = np.load(self.root + '/resized/' + filename[-27:-5] + '.npy')
        print(slice)
        print(data.shape[0] // 2 + slice)
        return data[data.shape[0] // 2 + slice]

    def __len__(self):
        return len(self.data)

class CTTestDataset(DataLoader):
    def __init__(self, root, size):
        self.root = root
        self.filenames = glob.glob(root + '*.hdf5')
        self.data = []

        if not os.path.exists(root + '/resized'):
            os.mkdir(root + '/resized')
            transform = v2.Compose([
                v2.RandomRotation([-90, -90]),
                v2.Resize((320, 320), antialias=True)
                ])
            for filename in self.filenames:
                img = transform(th.unsqueeze(th.from_numpy(h5py.File(filename, 'r')['data'][:, :, :]), 1))
                img = img - th.min(img)
                img = img / th.max(img)
                np.save(root + '/resized/' + filename[-27:-5], img.numpy())

        for filename in self.filenames:
            self.data.extend(select_slices(filename, 50))
        
        trial = open(f'./split/ct/test.txt', 'r').read()
        trial = ast.literal_eval(trial)
        self.data = trial

    def __getitem__(self, idx):
        filename = self.data[idx][0]
        slice = self.data[idx][1]
        data = np.load(self.root + '/resized/' + filename[-27:-5] + '.npy')
        # open(f'./split/ct/test.txt', 'a').write(f'{str(self.data[idx])}, ')
        return data[data.shape[0] // 2 + slice]

    def __len__(self):
        return len(self.data)

def create_ct_dataloader(size, batch_size=1):
    train_dataset = CTDataset('/srv/local/lg/ct_2d/', size)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_dataset = CTTestDataset('/srv/local/lg/ct_2d_test/', size)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True
    )
    return train_loader, test_loader

class CTHeadDataset(DataLoader):
    def __init__(self, root, size):
        self.root = root

        if not os.path.exists(root + 'resized'):
            self.filenames = glob.glob(root + '*.png')
            os.mkdir(root + 'resized')
            transform = v2.Compose([
                v2.Resize((320, 320), antialias=True)
                ])
            for i, filename in enumerate(self.filenames):
                img = transform(th.unsqueeze(th.unsqueeze(th.from_numpy(plt.imread(filename)[:, :, 0]), 0), 0))

                img = img - th.min(img)
                img = img / th.max(img)
                np.save(root + f'resized/img_{i}', img.numpy())
                plt.imsave(root + f'resized/img_{i}.png', img.squeeze().numpy(), cmap='gray')

        trial = open(f'./split/ct_head/test.txt', 'r').read()
        trial = ast.literal_eval(trial)
        self.filenames = trial
        
        self.data = self.filenames

    def __getitem__(self, idx):
        filename = self.data[idx]
        data = np.load(filename)
        return data

    def __len__(self):
        return len(self.data)

def create_ct_head_dataloader(size, batch_size=1):
    test_dataset = CTHeadDataset('/srv/local/lg/ct_brain/rsna-intracranial-hemorrhage-detection/png/', size)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True
    )
    return test_loader
