import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

save_root = Path('/media/lukasglaszner/data/figures')
save_root.mkdir(parents=True, exist_ok=True)
results_root = Path('/media/lukasglaszner/data/results')

aspect = (10, 10)

models = ['fastmri_knee_4', 'fastmri_knee_3', 'fastmri_knee_2', 'fastmri_knee_1']
legend_entries = [r'$d=4$', r'$d=3$', r'$d=2$', r'd=1']
recos = ['g1d4', 'g1d8', 'g2d4', 'r11', 'p15']
x_labels = ['Gaussian 1D 4x', 'Gaussian 1D 8x', 'Gaussian 2D 4x', 'Radial', 'Poisson 15x']
masks = [r'$\mathcal{N}$-1D', r'$\mathcal{N}$-1D', r'$\mathcal{N}$-2D', 'R', 'P']
acc = ['4', '8', '4', '11', '15']
Ns = [250]

psnr_fastmri = np.zeros((len(models), len(recos)))
ssim_fastmri = np.zeros((len(models), len(recos)))

g1d4_pc = np.loadtxt((results_root / 'mrncsn_fastmri_corpdfs_g1d4_pc' / 'psnr_values').with_suffix('.csv'), delimiter=',')
g1d8_pc = np.loadtxt((results_root / 'mrncsn_fastmri_corpdfs_g1d8_pc' / 'psnr_values').with_suffix('.csv'), delimiter=',')
g1d4_ald = np.loadtxt((results_root / 'mrncsn_fastmri_corpdfs_g1d4_ald' / 'psnr_values').with_suffix('.csv'), delimiter=',')
g1d8_ald = np.loadtxt((results_root / 'mrncsn_fastmri_corpdfs_g1d8_ald' / 'psnr_values').with_suffix('.csv'), delimiter=',')

plt.figure()
plt.plot(np.mean(g1d4_pc, axis=0)[::-1], color='C0', label='G-1D x4')
plt.plot(np.mean(g1d8_pc, axis=0)[::-1], color='C1', label='G-1D x8')
plt.plot(np.mean(g1d4_ald, axis=0)[::-1], color='C0', linestyle='--')
plt.plot(np.mean(g1d8_ald, axis=0)[::-1], color='C1', linestyle='--')
plt.xticks(np.arange(len(legend_entries)), legend_entries)
plt.ylabel('PSNR in dB')
plt.legend()
plt.savefig(save_root / 'psnr.pdf')
plt.close()

g1d4_pc = np.loadtxt((results_root / 'mrncsn_fastmri_corpdfs_g1d4_pc' / 'ssim_values').with_suffix('.csv'), delimiter=',')
g1d8_pc = np.loadtxt((results_root / 'mrncsn_fastmri_corpdfs_g1d8_pc' / 'ssim_values').with_suffix('.csv'), delimiter=',')
g1d4_ald = np.loadtxt((results_root / 'mrncsn_fastmri_corpdfs_g1d4_ald' / 'ssim_values').with_suffix('.csv'), delimiter=',')
g1d8_ald = np.loadtxt((results_root / 'mrncsn_fastmri_corpdfs_g1d8_ald' / 'ssim_values').with_suffix('.csv'), delimiter=',')
plt.figure()
plt.plot(np.mean(g1d4_pc, axis=0)[::-1], color='C0', label='G-1D x4')
plt.plot(np.mean(g1d8_pc, axis=0)[::-1], color='C1', label='G-1D x8')
plt.plot(np.mean(g1d4_ald, axis=0)[::-1], color='C0', linestyle='--')
plt.plot(np.mean(g1d8_ald, axis=0)[::-1], color='C1', linestyle='--')
plt.xticks(np.arange(len(legend_entries)), legend_entries)
plt.ylabel('SSIM in a.u.')
plt.legend()
plt.savefig(save_root / 'ssim.pdf')
plt.close()
