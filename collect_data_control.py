import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

root = Path('/mount/data/glaszner/ijcv_update')
masks = {'mri': ['g1d4', 'g1d8', 'g2d4', 'radial', 'poisson'], 'ct': ['sv60', 'sv30', 'sv20', 'fb60']}
datasets = {'mri': ['corpd', 'corpdfs', 'brain'], 'ct': ['thorax', 'head']}
algorithms = ['ald', 'pc', 'dps']
datatypes = {'mri': ['fastmri'], 'ct': ['ct']}
models = ['unet', 'tv']
methodologies = ['mri', 'ct']


for l, methodology in enumerate(methodologies):
    for datatype in datatypes[methodology]:
        # if methodology == 'mri' and algorithm != 'dps':
        #     continue
        for k, dataset in enumerate(datasets[methodology]):
            psnr_values = np.zeros((len(models), len(masks[methodology]) + 1))
            ssim_values = np.zeros((len(models), len(masks[methodology]) + 1))
            psnr_values[:, 0] = np.arange(len(models))
            ssim_values[:, 0] = np.arange(len(models))
            for j, model in enumerate(models):
                for i, mask in enumerate(masks[methodology]):
                    print(f'{datatype}, {model}, {dataset}, {mask}:')
                    try:
                        psnr = np.loadtxt((root / f'{model}_{datatype}_{dataset}_{mask}' / 'psnr_values').with_suffix('.csv'), delimiter=',')
                        ssim = np.loadtxt((root / f'{model}_{datatype}_{dataset}_{mask}' / 'ssim_values').with_suffix('.csv'), delimiter=',')
                        assert np.all(psnr) and np.all(np.isfinite(psnr))
                        assert np.all(ssim) and np.all(np.isfinite(ssim))
                    except FileNotFoundError:
                        print(f'No data')
                        continue
                    except AssertionError:
                        print(f'Invalid (NaN or Inf) or insufficient data')
                        continue
                    except Exception as e:
                        print(f'Unknown error: {e}')
                        continue
                    print(np.mean(psnr))
                    print(np.mean(ssim))
                    psnr_values[j, i + 1] = np.mean(psnr)
                    ssim_values[j, i + 1] = np.mean(ssim)
