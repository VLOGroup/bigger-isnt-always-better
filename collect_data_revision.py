import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# root = Path('/media/lukasglaszner/data/results_complete')
# masks = ['g1d4', 'g1d8', 'g2d4', 'radial', 'poisson']
# datasets = ['corpd', 'corpdfs', 'brain']
# algorithms = ['ald', 'pc', 'dps']
# models = ['fastmri', 'celeba']

# for model in models:
#     for algorithm in algorithms:
#         for dataset in datasets:
#             psnr_full = np.zeros((4, len(masks) + 1))
#             ssim_full = np.zeros((4, len(masks) + 1))
#             psnr_full[:, 0] = np.arange(4)
#             ssim_full[:, 0] = np.arange(4)
#             for i, mask in enumerate(masks):
#                 print(f'{algorithm}, {dataset}, {mask}:')
#                 try:
#                     psnr_values = np.loadtxt((root / f'mrncsn_structured_{model}_{dataset}_{mask}_{algorithm}' / 'psnr_values').with_suffix('.csv'), delimiter=',')
#                     ssim_values = np.loadtxt((root / f'mrncsn_structured_{model}_{dataset}_{mask}_{algorithm}' / 'ssim_values').with_suffix('.csv'), delimiter=',')
#                     assert np.all(psnr_values) and np.all(np.isfinite(psnr_values))
#                     assert np.all(ssim_values) and np.all(np.isfinite(ssim_values))
#                 except FileNotFoundError:
#                     print(f'No data')
#                     continue
#                 except AssertionError:
#                     print(f'Invalid (NaN or Inf) or insufficient data')
#                     continue
#                 except Exception as e:
#                     print(f'Unknown error: {e}')
#                     continue
#                 print(np.mean(psnr_values, axis=0))
#                 print(np.mean(ssim_values, axis=0))
#                 psnr_full[:, i + 1] = np.mean(psnr_values, axis=0)[::-1]
#                 ssim_full[:, i + 1] = np.mean(ssim_values, axis=0)[::-1]
#             if model == 'fastmri':
#                 np.savetxt(root / f'mrncsn_structured_{model}_{dataset}_{algorithm}_psnr.csv', psnr_full, delimiter=',')
#                 np.savetxt(root / f'mrncsn_structured_{model}_{dataset}_{algorithm}_ssim.csv', ssim_full, delimiter=',')
#             elif model == 'celeba':
#                 temp = np.loadtxt(root / f'mrncsn_structured_fastmri_{dataset}_{algorithm}_psnr.csv', delimiter=',')
#                 temp[:, 0] = 0
#                 np.savetxt(root / f'mrncsn_structured_{model}_{dataset}_{algorithm}_psnr.csv', psnr_full - temp, delimiter=',')
#                 temp = np.loadtxt(root / f'mrncsn_structured_fastmri_{dataset}_{algorithm}_ssim.csv', delimiter=',')
#                 temp[:, 0] = 0
#                 np.savetxt(root / f'mrncsn_structured_{model}_{dataset}_{algorithm}_ssim.csv', ssim_full - temp, delimiter=',')




# masks = ['sv60', 'sv30', 'sv20']
# datasets = ['thorax', 'head']
# algorithms = ['ald', 'pc', 'dps']

# for algorithm in algorithms:
#     for dataset in datasets:
#         for mask in masks:
#             print(f'{algorithm}, {dataset}, {mask}:')
#             try:
#                 psnr_values = np.loadtxt((root / f'mrncsn_structured_fastmri_{dataset}_{mask}_{algorithm}' / 'psnr_values').with_suffix('.csv'), delimiter=',')
#                 ssim_values = np.loadtxt((root / f'mrncsn_structured_fastmri_{dataset}_{mask}_{algorithm}' / 'ssim_values').with_suffix('.csv'), delimiter=',')
#                 assert np.all(psnr_values) and np.all(np.isfinite(psnr_values))
#                 assert np.all(ssim_values) and np.all(np.isfinite(ssim_values))
#             except FileNotFoundError:
#                 print(f'No data')
#                 continue
#             except AssertionError:
#                 print(f'Invalid (NaN or Inf) or insufficient data')
#                 continue
#             except Exception as e:
#                 print(f'Unknown error: {e}')
#                 continue
#             print(np.mean(psnr_values, axis=0))
#             print(np.mean(ssim_values, axis=0))


# psnr = {'corpd': np.zeros((49, 4)), 'corpdfs': np.zeros((49, 4)), 'brain': np.zeros((49, 4))}
# ssim = {'corpd': np.zeros((49, 4)), 'corpdfs': np.zeros((49, 4)), 'brain': np.zeros((49, 4))}
# dps_root = {}
# for k in psnr.keys():
#     dps_root.update({k: root / f'mrncsn_structured_fastmri_{k}_radial_dps_sde/intermediates'})

# print(dps_root)

# for i in range(100):
#     for d in range(4):
#         for dataset in psnr.keys():
#             psnr[dataset][:, d] += np.loadtxt(dps_root[dataset] / f'sample_{i}_d{d}' / 'psnr_values_x0_hat.txt', delimiter=',')
#             ssim[dataset][:, d] += np.loadtxt(dps_root[dataset] / f'sample_{i}_d{d}' / 'ssim_values_x.txt', delimiter=',')

# for p, s in zip(psnr.values(), ssim.values()):
#     p /= 100
#     s /= 100

# x = np.arange(0, 950, 25).reshape(-1, 1)
# x = np.vstack((x, np.arange(950, 1000, 5).reshape(-1, 1)))
# x = np.vstack((x, np.array([[999]])))
# print(x)
# plt.figure(figsize=(16,8))
# for i, dataset in enumerate(psnr.keys()):
#     for d in [0, 3]:
#         plt.plot(x, psnr[dataset][:, d], label=dataset, c=f'C{i}', ls='-' if d == 0 else '--')
# plt.xlim(500, 1000)
# plt.ylim(0, 40)
# plt.legend()
# plt.show()

root = Path('/mount/data/glaszner/ijcv_update')
masks = {'mri': ['gaussian_1d_4', 'gaussian_1d_8', 'gaussian_2d_4', 'radial', 'poisson'], 'ct': ['sparse_view_60', 'sparse_view_30', 'sparse_view_20', 'fb45']}
datasets = {'mri': ['corpd', 'corpdfs', 'brain'], 'ct': ['thorax', 'head']}
algorithms = ['ald', 'pc', 'dps']
datatypes = {'mri': ['fastmri', 'celeba'], 'ct': ['ct', 'celeba']}
models = ['4_small', '1_large']
methodologies = ['mri', 'ct']

celeba_percentages_psnr = np.zeros((5, 2))
celeba_percentages_ssim = np.zeros((5, 2))

for l, methodology in enumerate(methodologies):
    for datatype in datatypes[methodology]:
        for algorithm in algorithms:
            # if methodology == 'mri' and algorithm != 'dps':
            #     continue
            for k, dataset in enumerate(datasets[methodology]):
                psnr_values = np.zeros((len(models), len(masks[methodology]) + 1))
                ssim_values = np.zeros((len(models), len(masks[methodology]) + 1))
                psnr_values[:, 0] = np.arange(len(models))
                ssim_values[:, 0] = np.arange(len(models))
                for j, model in enumerate(models):
                    for i, mask in enumerate(masks[methodology]):
                        print(f'{datatype}, {model}, {dataset}, {mask}, {algorithm}:')
                        try:
                            psnr = np.loadtxt((root / f'{datatype}_{model}_{dataset}_{mask}_{algorithm}' / 'psnr_values').with_suffix('.csv'), delimiter=',')
                            ssim = np.loadtxt((root / f'{datatype}_{model}_{dataset}_{mask}_{algorithm}' / 'ssim_values').with_suffix('.csv'), delimiter=',')
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

                if datatype == 'celeba':
                    psnr_values[:, 1:] -= np.loadtxt(root / f'{"fastmri" if methodology == "mri" else "ct"}_{dataset}_{algorithm}_psnr.csv', delimiter=',')[:, 1:]
                    ssim_values[:, 1:] -= np.loadtxt(root / f'{"fastmri" if methodology == "mri" else "ct"}_{dataset}_{algorithm}_ssim.csv', delimiter=',')[:, 1:]
                    celeba_percentages_psnr[l * 3 + k, 0] += np.sum(psnr_values[:, 1:] > 0)
                    celeba_percentages_psnr[l * 3 + k, 1] += np.sum(np.ones_like(psnr_values[:, 1:]))
                    celeba_percentages_ssim[l * 3 + k, 0] += np.sum(ssim_values[:, 1:] > 0)
                    celeba_percentages_ssim[l * 3 + k, 1] += np.sum(np.ones_like(ssim_values[:, 1:]))
                np.savetxt(root / f'{datatype}_{dataset}_{algorithm}_psnr.csv', psnr_values, delimiter=',')
                np.savetxt(root / f'{datatype}_{dataset}_{algorithm}_ssim.csv', ssim_values, delimiter=',')


print(celeba_percentages_psnr[:, 0])
print(celeba_percentages_ssim[:, 0])
print(celeba_percentages_psnr[:, 0] / celeba_percentages_psnr[:, 1])
print(celeba_percentages_ssim[:, 0] / celeba_percentages_ssim[:, 1])

for algorithm in algorithms:
    psnr = np.loadtxt(root / f'ct_thorax_{algorithm}_psnr.csv', delimiter=',')
    print(psnr)
    for i in range(1, 4):
        print(np.polynomial.polynomial.polyfit(psnr[:, 0], psnr[:, i], 1))