import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

save_root = Path('./figures')
save_root.mkdir(parents=True, exist_ok=True)

aspect = (10, 10)

models = ['fastmri_knee_4_attention', 'fastmri_knee_4', 'fastmri_knee_3', 'fastmri_knee_2', 'fastmri_knee_1']
legend_entries = ['Depth = 4, Attention', 'Depth = 4', 'Depth = 3', 'Depth = 2', 'Depth = 1']
recos = ['CF_004', 'CF_004_AF_8', 'gaussian2d', 'radial', 'poisson_AF_15']
x_labels = ['Gaussian 1D 4x', 'Gaussian 1D 8x', 'Gaussian 2D 4x', 'Radial', 'Poisson 15x']
masks = [r'$\mathcal{N}$-1D', r'$\mathcal{N}$-1D', r'$\mathcal{N}$-2D', 'R', 'P']
acc = ['4', '8', '4', '11', '15']
Ns = [250]

psnr_fastmri = np.zeros((len(models), len(recos)))
ssim_fastmri = np.zeros((len(models), len(recos)))

positions = np.array([-0.2, -0.1, 0, 0.1, 0.2])
colors = ['pink', 'lightblue', 'lightgreen', 'moccasin', 'lightgray']
data = []
data_positions = []
patches = []

# ------ CORPD PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPD data set')
plt.tight_layout()
plt.savefig(save_root / 'psnr_corpd.pdf')
plt.close()
median_psnr_corpd = np.array([np.median(x) for x in data])
mean_psnr_corpd = np.array([np.mean(x) for x in data])
std_psnr_corpd = np.array([np.std(x) for x in data])

plt.figure()
plt.plot(np.reshape(mean_psnr_corpd, (len(models), len(recos))), linestyle='--', marker='.')
#plt.fill_between(np.arange(5), (np.reshape(mean_psnr_corpd, (len(models), len(recos))) - np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(mean_psnr_corpd, (len(models), len(recos))) + np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
plt.gca().set_xticks(range(len(models)))
plt.gca().set_xticklabels(legend_entries)
plt.legend(x_labels)
plt.ylabel('PSNR in dB')
plt.savefig(save_root / 'psnr_corpd_plot.pdf')
plt.close()


# ------ CORPDFS PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPDFS data set')
plt.tight_layout()
plt.savefig(save_root / 'psnr_corpdfs.pdf')
plt.close()
median_psnr_corpdfs = np.array([np.median(x) for x in data])
mean_psnr_corpdfs = np.array([np.mean(x) for x in data])
std_psnr_corpdfs = np.array([np.std(x) for x in data])


# ------ CORPD SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPD data set')
plt.tight_layout()
plt.savefig(save_root / 'ssim_corpd.pdf')
plt.close()
median_ssim_corpd = np.array([np.median(x) for x in data])
mean_ssim_corpd = np.array([np.mean(x) for x in data])
std_ssim_corpd = np.array([np.std(x) for x in data])


# ------ CORPDFS SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPDFS data set')
plt.tight_layout()
plt.savefig(save_root / 'ssim_corpdfs.pdf')
plt.close()
median_ssim_corpdfs = np.array([np.median(x) for x in data])
mean_ssim_corpdfs = np.array([np.mean(x) for x in data])
std_ssim_corpdfs = np.array([np.std(x) for x in data])

# ------ CORPD NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPD data set')
plt.tight_layout()
plt.savefig(save_root / 'nmse_corpd.pdf')
plt.close()


# ------ CORPDFS NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPDFS data set')
plt.tight_layout()
plt.savefig(save_root / 'nmse_corpdfs.pdf')
plt.close()


# # ------ Brain PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('Brain data set')
plt.tight_layout()
plt.savefig(save_root / 'psnr_brain.pdf')
plt.close()
median_psnr_brain = np.array([np.median(x) for x in data])
mean_psnr_brain = np.array([np.mean(x) for x in data])
std_psnr_brain = np.array([np.std(x) for x in data])


# # ------ Brain SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('Brain data set')
plt.tight_layout()
plt.savefig(save_root / 'ssim_brain.pdf')
plt.close()
median_ssim_brain = np.array([np.median(x) for x in data])
mean_ssim_brain = np.array([np.mean(x) for x in data])
std_ssim_brain = np.array([np.std(x) for x in data])

# # ------ Brain NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('Brain data set')
plt.tight_layout()
plt.savefig(save_root / 'nmse_brain.pdf')
plt.close()



#--------------------- CelebA-HQ -------------------------------
models = ['celeba_4_attention', 'celeba_4', 'celeba_3', 'celeba_2', 'celeba_1']
# ------ CORPD PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, median_psnr_corpd, c='red', marker='x', zorder=3, label='In-distribution median values')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPD data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_psnr_corpd.pdf')
plt.close()
mean_psnr_corpd_celeba = np.array([np.mean(x) for x in data])
std_psnr_corpd_celeba = np.array([np.std(x) for x in data])

plt.figure()
for i in range(len(recos)):
    plt.plot(np.reshape(mean_psnr_corpd, (len(models), len(recos)))[:, i], linestyle=':', marker='.', c=f'C{i}')
    plt.plot(np.reshape(mean_psnr_corpd_celeba, (len(models), len(recos)))[:, i], linestyle='--', marker='x', c=f'C{i}')
    plt.fill_between(np.arange(5), np.reshape(mean_psnr_corpd, (len(models), len(recos)))[:, i], np.reshape(mean_psnr_corpd_celeba, (len(models), len(recos)))[:, i], color=f'C{i}', alpha=0.3)
#plt.fill_between(np.arange(5), (np.reshape(mean_psnr_corpd, (len(models), len(recos))) - np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(mean_psnr_corpd, (len(models), len(recos))) + np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
plt.gca().set_xticks(range(len(models)))
plt.gca().set_xticklabels(legend_entries)
plt.legend(x_labels)
plt.ylabel('PSNR in dB')
plt.savefig(save_root / 'celeba_psnr_corpd_plot.pdf')
plt.close()


# ------ CORPDFS PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, median_psnr_corpdfs, c='red', marker='x', zorder=3, label='In-distribution median values')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPDFS data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_psnr_corpdfs.pdf')
plt.close()
mean_psnr_corpdfs_celeba = np.array([np.mean(x) for x in data])
std_psnr_corpdfs_celeba = np.array([np.std(x) for x in data])


# ------ CORPD SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, median_ssim_corpd, c='red', marker='x', zorder=3, label='In-distribution median values')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPD data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_ssim_corpd.pdf')
plt.close()
mean_ssim_corpd_celeba = np.array([np.mean(x) for x in data])
std_ssim_corpd_celeba = np.array([np.std(x) for x in data])


# ------ CORPDFS SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, median_ssim_corpdfs, c='red', marker='x', zorder=3, label='In-distribution means')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPDFS data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_ssim_corpdfs.pdf')
plt.close()
mean_ssim_corpdfs_celeba = np.array([np.mean(x) for x in data])
std_ssim_corpdfs_celeba = np.array([np.std(x) for x in data])

plt.figure()
for i in range(len(recos)):
    plt.plot(np.reshape(mean_ssim_corpdfs, (len(models), len(recos)))[:, i], linestyle=':', marker='.', c=f'C{i}')
    plt.plot(np.reshape(mean_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i], linestyle='--', marker='x', c=f'C{i}')
    plt.fill_between(np.arange(5), np.reshape(mean_ssim_corpdfs, (len(models), len(recos)))[:, i], np.reshape(mean_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i], color=f'C{i}', alpha=0.1)
#plt.fill_between(np.arange(5), (np.reshape(mean_psnr_corpd, (len(models), len(recos))) - np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(mean_psnr_corpd, (len(models), len(recos))) + np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
plt.gca().set_xticks(range(len(models)))
plt.gca().set_xticklabels([r'$d=4*$', r'$d=4$', r'$d=3$', r'$d=2$', r'$d=1$'])
plt.legend(x_labels)
plt.ylabel('SSIM in a.u.')
plt.savefig(save_root / 'celeba_ssim_corpdfs_plot.pdf')
plt.close()

# ------ CORPD NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPD data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_nmse_corpd.pdf')
plt.close()


# ------ CORPDFS NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPDFS data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_nmse_corpdfs.pdf')
plt.close()

# # ------ Brain PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, median_psnr_brain, c='red', marker='x', zorder=3, label='In-distribution median values')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('Brain data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_psnr_brain.pdf')
plt.close()
mean_psnr_brain_celeba = np.array([np.mean(x) for x in data])
std_psnr_brain_celeba = np.array([np.std(x) for x in data])


# # ------ Brain SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, median_ssim_brain, c='red', marker='x', zorder=3, label='In-distribution median values')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('Brain data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_ssim_brain.pdf')
plt.close()
mean_ssim_brain_celeba = np.array([np.mean(x) for x in data])
std_ssim_brain_celeba = np.array([np.std(x) for x in data])

# # ------ Brain NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('Brain data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_nmse_brain.pdf')
plt.close()

psnr = [mean_psnr_corpd, mean_psnr_corpdfs, mean_psnr_brain]
psnr_celeba = [mean_psnr_corpd_celeba, mean_psnr_corpdfs_celeba, mean_psnr_brain_celeba]
ssim = [mean_ssim_corpd, mean_ssim_corpdfs, mean_ssim_brain]
ssim_celeba = [mean_ssim_corpd_celeba, mean_ssim_corpdfs_celeba, mean_ssim_brain_celeba]


# Tables
print(r'\sisetup{table-alignment-mode = format, table-number-alignment = center}'
      r'\begin{tabular}{ccc*{5}{S[table-format=2.2(1),separate-uncertainty,table-align-uncertainty,detect-weight]}}\toprule')
print(r' & & {AF} &' + ' & '.join([r'{$d=4*$}', r'{$d=4$}', r'{$d=3$}', r'{$d=2$}', r'{$d=1$}']) + r'\\\midrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{CORPD}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(mean_psnr_corpd, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(mean_ssim_corpd, (len(models), len(recos)))[:, i])
    print(' &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(mean_psnr_corpd, (len(models), len(recos)))[:, i], np.reshape(std_psnr_corpd, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(' & & &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(mean_ssim_corpd, (len(models), len(recos)))[:, i], np.reshape(std_ssim_corpd, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\midrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{CORPDFS}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(mean_psnr_corpdfs, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(mean_ssim_corpdfs, (len(models), len(recos)))[:, i])
    print(' &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(mean_psnr_corpdfs, (len(models), len(recos)))[:, i], np.reshape(std_psnr_corpdfs, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(' & & &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(mean_ssim_corpdfs, (len(models), len(recos)))[:, i], np.reshape(std_ssim_corpdfs, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\midrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{Brain}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(mean_psnr_brain, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(mean_ssim_brain, (len(models), len(recos)))[:, i])
    print(' &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(mean_psnr_brain, (len(models), len(recos)))[:, i], np.reshape(std_psnr_brain, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(' & & &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(mean_ssim_brain, (len(models), len(recos)))[:, i], np.reshape(std_ssim_brain, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\bottomrule')
print(r'\end{tabular}')



fig = plt.figure(figsize=(15, 12), layout='constrained')
subfigs = fig.subfigures(ncols=1, nrows=3)
axs = []
for i in range(3):
    axs.append(subfigs[i].subplots(ncols=2, nrows=1))
for j in range(3):
    for i in range(len(recos)):
        axs[j][0].plot(np.reshape(psnr[j], (len(models), len(recos)))[:, i], linestyle='--', marker='x', c=f'C{i}', label=('fastMRI – ' + x_labels[i]))
    #plt.fill_between(np.arange(5), (np.reshape(mean_psnr_corpd, (len(models), len(recos))) - np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(mean_psnr_corpd, (len(models), len(recos))) + np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
    axs[j][0].set_xticks(range(len(models)))
    axs[j][0].set_xticklabels([r'$d=4*$', r'$d=4$', r'$d=3$', r'$d=2$', r'$d=1$'])
    if j == 0:
        axs[j][0].legend(loc=3)
    # axs[j, 0].legend(x_labels)
    axs[j][0].set_ylabel('PSNR in dB')
    axs[j][0].set_ylim((28, 34))
for j in range(3):
    for i in range(len(recos)):
        axs[j][1].plot(np.reshape(ssim[j], (len(models), len(recos)))[:, i], linestyle='--', marker='x', c=f'C{i}')
    #plt.fill_between(np.arange(5), (np.reshape(mean_psnr_corpd, (len(models), len(recos))) - np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(mean_psnr_corpd, (len(models), len(recos))) + np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
    axs[j][1].set_xticks(range(len(models)))
    axs[j][1].set_xticklabels([r'$d=4*$', r'$d=4$', r'$d=3$', r'$d=2$', r'$d=1$'])
    # axs[j, 1].legend(x_labels)
    axs[j][1].set_ylabel('SSIM in a.u.')
    axs[j][1].set_ylim((0.55, 0.9))
#fig.tight_layout()
subfigs[0].suptitle('CORPD data set')
subfigs[1].suptitle('CORPDFS data set')
subfigs[2].suptitle('Brain data set')
plt.savefig(save_root / 'fastmri.pdf')
plt.close()


fig = plt.figure(figsize=(15, 12), layout='constrained')
subfigs = fig.subfigures(ncols=1, nrows=3)
axs = []
for i in range(3):
    axs.append(subfigs[i].subplots(ncols=2, nrows=1))
for j in range(3):
    for i in range(len(recos)):
        axs[j][0].plot(np.reshape(psnr[j], (len(models), len(recos)))[:, i], linestyle=':', marker='.', c=f'C{i}')
        axs[j][0].plot(np.reshape(psnr_celeba[j], (len(models), len(recos)))[:, i], linestyle='--', marker='x', c=f'C{i}', label=(r'CelebA-HQ – ' + x_labels[i]))
        axs[j][0].fill_between(np.arange(5), np.reshape(psnr[j], (len(models), len(recos)))[:, i], np.reshape(psnr_celeba[j], (len(models), len(recos)))[:, i], color=f'C{i}', linewidth=0, alpha=0.075)
    #plt.fill_between(np.arange(5), (np.reshape(mean_psnr_corpd, (len(models), len(recos))) - np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(mean_psnr_corpd, (len(models), len(recos))) + np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
    axs[j][0].set_xticks(range(len(models)))
    axs[j][0].set_xticklabels([r'$d=4*$', r'$d=4$', r'$d=3$', r'$d=2$', r'$d=1$'])
    if j == 0:
        axs[j][0].legend(loc=3)
    # axs[j, 0].legend(x_labels)
    axs[j][0].set_ylabel('PSNR in dB')
for j in range(3):
    for i in range(len(recos)):
        axs[j][1].plot(np.reshape(ssim[j], (len(models), len(recos)))[:, i], linestyle=':', marker='.', c=f'C{i}', label=('fastMRI – ' + x_labels[i]))
        axs[j][1].plot(np.reshape(ssim_celeba[j], (len(models), len(recos)))[:, i], linestyle='--', marker='x', c=f'C{i}')
        axs[j][1].fill_between(np.arange(5), np.reshape(ssim[j], (len(models), len(recos)))[:, i], np.reshape(ssim_celeba[j], (len(models), len(recos)))[:, i], color=f'C{i}', linewidth=0, alpha=0.075)
    #plt.fill_between(np.arange(5), (np.reshape(mean_psnr_corpd, (len(models), len(recos))) - np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(mean_psnr_corpd, (len(models), len(recos))) + np.reshape(std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
    axs[j][1].set_xticks(range(len(models)))
    axs[j][1].set_xticklabels([r'$d=4*$', r'$d=4$', r'$d=3$', r'$d=2$', r'$d=1$'])
    if j == 0:
        axs[j][1].legend(loc=3)
    # axs[j, 1].legend(x_labels)
    axs[j][1].set_ylabel('SSIM in a.u.')
#fig.tight_layout()
subfigs[0].suptitle('CORPD data set')
subfigs[1].suptitle('CORPDFS data set')
subfigs[2].suptitle('Brain data set')
plt.savefig(save_root / 'celeba.pdf')
plt.close()


save_root = Path('./figures_jalal')
save_root.mkdir(parents=True, exist_ok=True)

aspect = (10, 10)
N_slices = 100

path = 'jalal/jalal/'

models = ['fastmri_knee_4_attention', 'fastmri_knee_4', 'fastmri_knee_3', 'fastmri_knee_2', 'fastmri_knee_1']
legend_entries = ['Depth = 4, Attention', 'Depth = 4', 'Depth = 3', 'Depth = 2', 'Depth = 1']
recos = ['CF_004', 'CF_004_AF_8', 'gaussian2d', 'radial', 'poisson_AF_15']
x_labels = ['Gaussian 1D 4x', 'Gaussian 1D 8x', 'Gaussian 2D 4x', 'Radial', 'Poisson 15x']

acc = ['4', '8', '4', '11', '15']
Ns = [500]

psnr_fastmri = np.zeros((len(models), len(recos)))
ssim_fastmri = np.zeros((len(models), len(recos)))

positions = np.array([-0.2, -0.1, 0, 0.1, 0.2])
colors = ['pink', 'lightblue', 'lightgreen', 'moccasin', 'lightgray']
data = []
data_positions = []
patches = []

# ------ CORPD PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/psnr_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPD data set')
plt.tight_layout()
plt.savefig(save_root / 'psnr_corpd.pdf')
plt.close()
jalal_median_psnr_corpd = np.array([np.median(x) for x in data])
jalal_mean_psnr_corpd = np.array([np.mean(x) for x in data])
jalal_std_psnr_corpd = np.array([np.std(x) for x in data])

plt.figure()
plt.plot(np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))), linestyle='--', marker='.')
#plt.fill_between(np.arange(5), (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) - np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) + np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
plt.gca().set_xticks(range(len(models)))
plt.gca().set_xticklabels(legend_entries)
plt.legend(x_labels)
plt.ylabel('PSNR in dB')
plt.savefig(save_root / 'psnr_corpd_plot.pdf')
plt.close()


# ------ CORPDFS PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/psnr_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPDFS data set')
plt.tight_layout()
plt.savefig(save_root / 'psnr_corpdfs.pdf')
plt.close()
jalal_median_psnr_corpdfs = np.array([np.median(x) for x in data])
jalal_mean_psnr_corpdfs = np.array([np.mean(x) for x in data])
jalal_std_psnr_corpdfs = np.array([np.std(x) for x in data])


# ------ CORPD SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/ssim_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPD data set')
plt.tight_layout()
plt.savefig(save_root / 'ssim_corpd.pdf')
plt.close()
jalal_median_ssim_corpd = np.array([np.median(x) for x in data])
jalal_mean_ssim_corpd = np.array([np.mean(x) for x in data])
jalal_std_ssim_corpd = np.array([np.std(x) for x in data])


# ------ CORPDFS SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/ssim_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPDFS data set')
plt.tight_layout()
plt.savefig(save_root / 'ssim_corpdfs.pdf')
plt.close()
jalal_median_ssim_corpdfs = np.array([np.median(x) for x in data])
jalal_mean_ssim_corpdfs = np.array([np.mean(x) for x in data])
jalal_std_ssim_corpdfs = np.array([np.std(x) for x in data])

# ------ CORPD NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/nmse_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPD data set')
plt.tight_layout()
plt.savefig(save_root / 'nmse_corpd.pdf')
plt.close()


# ------ CORPDFS NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/nmse_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPDFS data set')
plt.tight_layout()
plt.savefig(save_root / 'nmse_corpdfs.pdf')
plt.close()


# # ------ Brain PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/psnr_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('Brain data set')
plt.tight_layout()
plt.savefig(save_root / 'psnr_brain.pdf')
plt.close()
jalal_median_psnr_brain = np.array([np.median(x) for x in data])
jalal_mean_psnr_brain = np.array([np.mean(x) for x in data])
jalal_std_psnr_brain = np.array([np.std(x) for x in data])


# # ------ Brain SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/ssim_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('Brain data set')
plt.tight_layout()
plt.savefig(save_root / 'ssim_brain.pdf')
plt.close()
jalal_median_ssim_brain = np.array([np.median(x) for x in data])
jalal_mean_ssim_brain = np.array([np.mean(x) for x in data])
jalal_std_ssim_brain = np.array([np.std(x) for x in data])

# # ------ Brain NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/nmse_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('Brain data set')
plt.tight_layout()
plt.savefig(save_root / 'nmse_brain.pdf')
plt.close()



#--------------------- CelebA-HQ -------------------------------
models = ['celeba_4_attention', 'celeba_4', 'celeba_3', 'celeba_2', 'celeba_1']
# ------ CORPD PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/psnr_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, jalal_median_psnr_corpd, c='red', marker='x', zorder=3, label='In-distribution median values')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPD data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_psnr_corpd.pdf')
plt.close()
jalal_mean_psnr_corpd_celeba = np.array([np.mean(x) for x in data])
jalal_std_psnr_corpd_celeba = np.array([np.std(x) for x in data])

plt.figure()
for i in range(len(recos)):
    plt.plot(np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos)))[:, i], linestyle=':', marker='.', c=f'C{i}')
    plt.plot(np.reshape(jalal_mean_psnr_corpd_celeba, (len(models), len(recos)))[:, i], linestyle='--', marker='x', c=f'C{i}')
    plt.fill_between(np.arange(5), np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos)))[:, i], np.reshape(jalal_mean_psnr_corpd_celeba, (len(models), len(recos)))[:, i], color=f'C{i}', alpha=0.3)
#plt.fill_between(np.arange(5), (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) - np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) + np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
plt.gca().set_xticks(range(len(models)))
plt.gca().set_xticklabels(legend_entries)
plt.legend(x_labels)
plt.ylabel('PSNR in dB')
plt.savefig(save_root / 'celeba_psnr_corpd_plot.pdf')
plt.close()


# ------ CORPDFS PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/psnr_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, jalal_median_psnr_corpdfs, c='red', marker='x', zorder=3, label='In-distribution median values')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPDFS data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_psnr_corpdfs.pdf')
plt.close()
jalal_mean_psnr_corpdfs_celeba = np.array([np.mean(x) for x in data])
jalal_std_psnr_corpdfs_celeba = np.array([np.std(x) for x in data])


# ------ CORPD SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/ssim_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, jalal_median_ssim_corpd, c='red', marker='x', zorder=3, label='In-distribution median values')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPD data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_ssim_corpd.pdf')
plt.close()
jalal_mean_ssim_corpd_celeba = np.array([np.mean(x) for x in data])
jalal_std_ssim_corpd_celeba = np.array([np.std(x) for x in data])


# ------ CORPDFS SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/ssim_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, jalal_median_ssim_corpdfs, c='red', marker='x', zorder=3, label='In-distribution means')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPDFS data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_ssim_corpdfs.pdf')
plt.close()
jalal_mean_ssim_corpdfs_celeba = np.array([np.mean(x) for x in data])
jalal_std_ssim_corpdfs_celeba = np.array([np.std(x) for x in data])

plt.figure()
for i in range(len(recos)):
    plt.plot(np.reshape(jalal_mean_ssim_corpdfs, (len(models), len(recos)))[:, i], linestyle=':', marker='.', c=f'C{i}')
    plt.plot(np.reshape(jalal_mean_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i], linestyle='--', marker='x', c=f'C{i}')
    plt.fill_between(np.arange(5), np.reshape(jalal_mean_ssim_corpdfs, (len(models), len(recos)))[:, i], np.reshape(jalal_mean_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i], color=f'C{i}', alpha=0.1)
#plt.fill_between(np.arange(5), (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) - np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) + np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
plt.gca().set_xticks(range(len(models)))
plt.gca().set_xticklabels([r'$d=4*$', r'$d=4$', r'$d=3$', r'$d=2$', r'$d=1$'])
plt.legend(x_labels)
plt.ylabel('SSIM in a.u.')
plt.savefig(save_root / 'celeba_ssim_corpdfs_plot.pdf')
plt.close()

# ------ CORPD NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/nmse_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPD data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_nmse_corpd.pdf')
plt.close()


# ------ CORPDFS NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/nmse_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPDFS data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_nmse_corpdfs.pdf')
plt.close()

# # ------ Brain PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/psnr_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, jalal_median_psnr_brain, c='red', marker='x', zorder=3, label='In-distribution median values')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('Brain data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_psnr_brain.pdf')
plt.close()
jalal_mean_psnr_brain_celeba = np.array([np.mean(x) for x in data])
jalal_std_psnr_brain_celeba = np.array([np.std(x) for x in data])


# # ------ Brain SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/ssim_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, jalal_median_ssim_brain, c='red', marker='x', zorder=3, label='In-distribution median values')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('Brain data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_ssim_brain.pdf')
plt.close()
jalal_mean_ssim_brain_celeba = np.array([np.mean(x) for x in data])
jalal_std_ssim_brain_celeba = np.array([np.std(x) for x in data])

# # ------ Brain NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt(path + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/nmse_values.csv', delimiter=',')[:N_slices])
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('Brain data set, out-of-distribution training')
plt.tight_layout()
plt.savefig(save_root / 'celeba_nmse_brain.pdf')
plt.close()

jalal_psnr = [jalal_mean_psnr_corpd, jalal_mean_psnr_corpdfs, jalal_mean_psnr_brain]
jalal_celeba_psnr = [jalal_mean_psnr_corpd_celeba, jalal_mean_psnr_corpdfs_celeba, jalal_mean_psnr_brain_celeba]
jalal_ssim = [jalal_mean_ssim_corpd, jalal_mean_ssim_corpdfs, jalal_mean_ssim_brain]
jalal_celeba_ssim = [jalal_mean_ssim_corpd_celeba, jalal_mean_ssim_corpdfs_celeba, jalal_mean_ssim_brain_celeba]


# Tables
print(r'\sisetup{table-alignment-mode = format, table-number-alignment = center, mode = text}'
      r'\begin{tabular}{ccc*{5}{S[table-format=2.2(1),separate-uncertainty,table-align-uncertainty,detect-weight]}}\toprule')
print(r' & & {AF} &' + ' & '.join([r'{$d=4*$}', r'{$d=4$}', r'{$d=3$}', r'{$d=2$}', r'{$d=1$}']) + r'\\\midrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{CORPD}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_corpd, (len(models), len(recos)))[:, i])
    print(' &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_corpd, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(' & & &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_ssim_corpd, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_corpd, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\midrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{CORPDFS}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_corpdfs, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_corpdfs, (len(models), len(recos)))[:, i])
    print(' &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_psnr_corpdfs, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_corpdfs, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(' & & &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_ssim_corpdfs, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_corpdfs, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\midrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{Brain}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_brain, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_brain, (len(models), len(recos)))[:, i])
    print(' &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_psnr_brain, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_brain, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(' & & &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_ssim_brain, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_brain, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\bottomrule')
print(r'\end{tabular}')

print('\n\n-----------------------------------------\n\n')

# Tables
print(r'\sisetup{table-alignment-mode = format, table-number-alignment = center, mode = text}'
      r'\begin{tabular}{ccc*{5}{S[table-format=2.2(1),separate-uncertainty,table-align-uncertainty,detect-weight]}}\toprule')
print(r' & & {AF} &' + ' & '.join([r'{$d=4*$}', r'{$d=4$}', r'{$d=3$}', r'{$d=2$}', r'{$d=1$}']) + r'\\\midrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{CORPD}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_corpd_celeba, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_corpd_celeba, (len(models), len(recos)))[:, i])
    print(' &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_psnr_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_corpd_celeba, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(' & & &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_ssim_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_corpd_celeba, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\midrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{CORPDFS}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_corpdfs_celeba, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i])
    print(' &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_psnr_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_corpdfs_celeba, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(' & & &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\midrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{Brain}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_brain_celeba, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_brain_celeba, (len(models), len(recos)))[:, i])
    print(' &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_psnr_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_brain_celeba, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(' & & &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_ssim_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_brain_celeba, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\bottomrule')
print(r'\end{tabular}')




fig = plt.figure(figsize=(15, 12), layout='constrained')
subfigs = fig.subfigures(ncols=1, nrows=3)
axs = []
for i in range(3):
    axs.append(subfigs[i].subplots(ncols=2, nrows=1))
for j in range(3):
    for i in range(len(recos)):
        axs[j][0].plot(np.reshape(jalal_psnr[j], (len(models), len(recos)))[:, i], linestyle='--', marker='x', c=f'C{i}', label=('fastMRI – ' + x_labels[i]))
    #plt.fill_between(np.arange(5), (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) - np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) + np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
    axs[j][0].set_xticks(range(len(models)))
    axs[j][0].set_xticklabels([r'$d=4*$', r'$d=4$', r'$d=3$', r'$d=2$', r'$d=1$'])
    if j == 0:
        axs[j][0].legend(loc=3)
    # axs[j, 0].legend(x_labels)
    axs[j][0].set_ylabel('PSNR in dB')
    axs[j][0].set_ylim((28, 34))
for j in range(3):
    for i in range(len(recos)):
        axs[j][1].plot(np.reshape(jalal_ssim[j], (len(models), len(recos)))[:, i], linestyle='--', marker='x', c=f'C{i}')
    #plt.fill_between(np.arange(5), (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) - np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) + np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
    axs[j][1].set_xticks(range(len(models)))
    axs[j][1].set_xticklabels([r'$d=4*$', r'$d=4$', r'$d=3$', r'$d=2$', r'$d=1$'])
    # axs[j, 1].legend(x_labels)
    axs[j][1].set_ylabel('SSIM in a.u.')
    axs[j][1].set_ylim((0.55, 0.9))
#fig.tight_layout()
subfigs[0].suptitle('CORPD data set')
subfigs[1].suptitle('CORPDFS data set')
subfigs[2].suptitle('Brain data set')
plt.savefig(save_root / 'fastmri.pdf')
plt.close()


fig = plt.figure(figsize=(15, 12), layout='constrained')
subfigs = fig.subfigures(ncols=1, nrows=3)
axs = []
for i in range(3):
    axs.append(subfigs[i].subplots(ncols=2, nrows=1))
for j in range(3):
    for i in range(len(recos)):
        axs[j][0].plot(np.reshape(jalal_psnr[j], (len(models), len(recos)))[:, i], linestyle=':', marker='.', c=f'C{i}')
        axs[j][0].plot(np.reshape(jalal_celeba_psnr[j], (len(models), len(recos)))[:, i], linestyle='--', marker='x', c=f'C{i}', label=(r'CelebA-HQ – ' + x_labels[i]))
        axs[j][0].fill_between(np.arange(5), np.reshape(jalal_psnr[j], (len(models), len(recos)))[:, i], np.reshape(jalal_celeba_psnr[j], (len(models), len(recos)))[:, i], color=f'C{i}', linewidth=0, alpha=0.075)
    #plt.fill_between(np.arange(5), (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) - np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) + np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
    axs[j][0].set_xticks(range(len(models)))
    axs[j][0].set_xticklabels([r'$d=4*$', r'$d=4$', r'$d=3$', r'$d=2$', r'$d=1$'])
    if j == 0:
        axs[j][0].legend(loc=3)
    # axs[j, 0].legend(x_labels)
    axs[j][0].set_ylabel('PSNR in dB')
for j in range(3):
    for i in range(len(recos)):
        axs[j][1].plot(np.reshape(jalal_ssim[j], (len(models), len(recos)))[:, i], linestyle=':', marker='.', c=f'C{i}', label=('fastMRI – ' + x_labels[i]))
        axs[j][1].plot(np.reshape(jalal_celeba_ssim[j], (len(models), len(recos)))[:, i], linestyle='--', marker='x', c=f'C{i}')
        axs[j][1].fill_between(np.arange(5), np.reshape(jalal_ssim[j], (len(models), len(recos)))[:, i], np.reshape(jalal_celeba_ssim[j], (len(models), len(recos)))[:, i], color=f'C{i}', linewidth=0, alpha=0.075)
    #plt.fill_between(np.arange(5), (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) - np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], (np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos))) + np.reshape(jalal_std_psnr_corpd, (len(models), len(recos))))[0, :], alpha=0.5)
    axs[j][1].set_xticks(range(len(models)))
    axs[j][1].set_xticklabels([r'$d=4*$', r'$d=4$', r'$d=3$', r'$d=2$', r'$d=1$'])
    if j == 0:
        axs[j][1].legend(loc=3)
    # axs[j, 1].legend(x_labels)
    axs[j][1].set_ylabel('SSIM in a.u.')
#fig.tight_layout()
subfigs[0].suptitle('CORPD data set')
subfigs[1].suptitle('CORPDFS data set')
subfigs[2].suptitle('Brain data set')
plt.savefig(save_root / 'celeba.pdf')
plt.close()




# Tables
print(r'\sisetup{table-alignment-mode = format, table-number-alignment = center, mode = text}'
      r'\begin{tabular}{ccc*{5}{S[table-format=2.2(1),separate-uncertainty,table-align-uncertainty,detect-weight]}}\toprule')
print(r' & & {AF} &' + ' & '.join([r'{$d=4*$}', r'{$d=4$}', r'{$d=3$}', r'{$d=2$}', r'{$d=1$}']) + r'\\\midrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{CORPD}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_corpd, (len(models), len(recos)))[:, i])
    print(' &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_corpd, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(' & & &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_ssim_corpd, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_corpd, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\midrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{CORPDFS}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_corpdfs, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_corpdfs, (len(models), len(recos)))[:, i])
    print(' &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_psnr_corpdfs, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_corpdfs, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(' & & &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_ssim_corpdfs, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_corpdfs, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\midrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{Brain}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_brain, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_brain, (len(models), len(recos)))[:, i])
    print(' &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_psnr_brain, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_brain, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(' & & &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_ssim_brain, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_brain, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\bottomrule')
print(r'\end{tabular}')

print('\n\n-----------------------------------------\n\n')

tv_arg_corpd = [np.argmax(np.mean(np.loadtxt(f'results/tv_{reco}_N_250_no_norm/psnr_values.csv', delimiter=','), axis=0)) for reco in recos]
tv_arg_corpdfs = [np.argmax(np.mean(np.loadtxt(f'results/tv_{reco}_N_250_FS_no_norm/psnr_values.csv', delimiter=','), axis=0)) for reco in recos]
tv_arg_brain = [np.argmax(np.mean(np.loadtxt(f'results/tv_{reco}_N_250_brain_no_norm/psnr_values.csv', delimiter=','), axis=0)) for reco in recos]
tv_psnr_corpd = [[np.mean(np.loadtxt(f'results/tv_{reco}_N_250_no_norm/psnr_values.csv', delimiter=','), axis=0)[i], np.std(np.loadtxt(f'results/tv_{reco}_N_250_no_norm/psnr_values.csv', delimiter=','), axis=0)[i]] for i, reco in zip(tv_arg_corpd, recos)]
tv_psnr_corpdfs = [[np.mean(np.loadtxt(f'results/tv_{reco}_N_250_FS_no_norm/psnr_values.csv', delimiter=','), axis=0)[i], np.std(np.loadtxt(f'results/tv_{reco}_N_250_FS_no_norm/psnr_values.csv', delimiter=','), axis=0)[i]] for i, reco in zip(tv_arg_corpdfs, recos)]
tv_psnr_brain = [[np.mean(np.loadtxt(f'results/tv_{reco}_N_250_brain_no_norm/psnr_values.csv', delimiter=','), axis=0)[i], np.std(np.loadtxt(f'results/tv_{reco}_N_250_brain_no_norm/psnr_values.csv', delimiter=','), axis=0)[i]] for i, reco in zip(tv_arg_brain, recos)]
tv_ssim_corpd = [[np.mean(np.loadtxt(f'results/tv_{reco}_N_250_no_norm/ssim_values.csv', delimiter=','), axis=0)[i], np.std(np.loadtxt(f'results/tv_{reco}_N_250_no_norm/ssim_values.csv', delimiter=','), axis=0)[i]] for i, reco in zip(tv_arg_corpd, recos)]
tv_ssim_corpdfs = [[np.mean(np.loadtxt(f'results/tv_{reco}_N_250_FS_no_norm/ssim_values.csv', delimiter=','), axis=0)[i], np.std(np.loadtxt(f'results/tv_{reco}_N_250_FS_no_norm/ssim_values.csv', delimiter=','), axis=0)[i]] for i, reco in zip(tv_arg_corpdfs, recos)]
tv_ssim_brain = [[np.mean(np.loadtxt(f'results/tv_{reco}_N_250_brain_no_norm/ssim_values.csv', delimiter=','), axis=0)[i], np.std(np.loadtxt(f'results/tv_{reco}_N_250_brain_no_norm/ssim_values.csv', delimiter=','), axis=0)[i]] for i, reco in zip(tv_arg_brain, recos)]
print(tv_arg_corpd)

unet_psnr_corpd = [[np.mean(np.loadtxt(f'results/unet_{reco}_N_250_no_norm/psnr_values.csv', delimiter=','), axis=0), np.std(np.loadtxt(f'results/unet_{reco}_N_250_no_norm/psnr_values.csv', delimiter=','), axis=0)] for reco in recos]
unet_psnr_corpdfs = [[np.mean(np.loadtxt(f'results/unet_{reco}_N_250_FS_no_norm/psnr_values.csv', delimiter=','), axis=0), np.std(np.loadtxt(f'results/unet_{reco}_N_250_FS_no_norm/psnr_values.csv', delimiter=','), axis=0)] for reco in recos]
unet_psnr_brain = [[np.mean(np.loadtxt(f'results/unet_{reco}_N_250_brain_no_norm/psnr_values.csv', delimiter=','), axis=0), np.std(np.loadtxt(f'results/unet_{reco}_N_250_brain_no_norm/psnr_values.csv', delimiter=','), axis=0)] for reco in recos]
unet_ssim_corpd = [[np.mean(np.loadtxt(f'results/unet_{reco}_N_250_no_norm/ssim_values.csv', delimiter=','), axis=0), np.std(np.loadtxt(f'results/unet_{reco}_N_250_no_norm/ssim_values.csv', delimiter=','), axis=0)] for reco in recos]
unet_ssim_corpdfs = [[np.mean(np.loadtxt(f'results/unet_{reco}_N_250_FS_no_norm/ssim_values.csv', delimiter=','), axis=0), np.std(np.loadtxt(f'results/unet_{reco}_N_250_FS_no_norm/ssim_values.csv', delimiter=','), axis=0)] for reco in recos]
unet_ssim_brain = [[np.mean(np.loadtxt(f'results/unet_{reco}_N_250_brain_no_norm/ssim_values.csv', delimiter=','), axis=0), np.std(np.loadtxt(f'results/unet_{reco}_N_250_brain_no_norm/ssim_values.csv', delimiter=','), axis=0)] for reco in recos]


# Tables
print(r'\sisetup{table-alignment-mode = format, table-number-alignment = center, mode = text}'
      r'\begin{tabular}{ccc*{2}{S[table-format=2.2(1),separate-uncertainty,table-align-uncertainty,detect-weight]}*{5}{S[table-format=2.2(1),separate-uncertainty,table-align-uncertainty,detect-weight]}*{5}{S[table-format=2.2(1),separate-uncertainty,table-align-uncertainty,detect-weight]}}\toprule')
print(r' & & & & & \multicolumn{5}{c}{Chung \& Ye} & \multicolumn{5}{c}{Jalal~\etal}\\')
print(r' & & {AF} & {TV} & {U-Net}' + (' & ' + ' & '.join([r'{$d=4*$}', r'{$d=4$}', r'{$d=3$}', r'{$d=2$}', r'{$d=1$}'])) * 2 + r'\\\mymidrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{CORPD}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(mean_psnr_corpd, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(mean_ssim_corpd, (len(models), len(recos)))[:, i])
    jalal_argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos)))[:, i])
    jalal_argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_corpd, (len(models), len(recos)))[:, i])
    print(f' & {tv_psnr_corpd[i][0]:.2f} \pm {tv_psnr_corpd[i][1]:.2f} & {unet_psnr_corpd[i][0]:.2f} \pm {unet_psnr_corpd[i][1]:.2f} &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(mean_psnr_corpd, (len(models), len(recos)))[:, i], np.reshape(std_psnr_corpd, (len(models), len(recos)))[:, i]))]))
    print(' & ' + ' & '.join([(r'\bfseries ' if j == jalal_argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_corpd, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(f' & & & {tv_ssim_corpd[i][0]:.2f} \pm {tv_ssim_corpd[i][1]:.2f} & {unet_ssim_corpd[i][0]:.2f} \pm {unet_ssim_corpd[i][1]:.2f} &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(mean_ssim_corpd, (len(models), len(recos)))[:, i], np.reshape(std_ssim_corpd, (len(models), len(recos)))[:, i]))]))
    print(' &' + ' & '.join([(r'\bfseries ' if j == jalal_argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_ssim_corpd, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_corpd, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\mymidrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{CORPDFS}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(mean_psnr_corpdfs, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(mean_ssim_corpdfs, (len(models), len(recos)))[:, i])
    jalal_argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_corpdfs, (len(models), len(recos)))[:, i])
    jalal_argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_corpdfs, (len(models), len(recos)))[:, i])
    print(f' & {tv_psnr_corpdfs[i][0]:.2f} \pm {tv_psnr_corpdfs[i][1]:.2f} & {unet_psnr_corpdfs[i][0]:.2f} \pm {unet_psnr_corpdfs[i][1]:.2f} &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(mean_psnr_corpdfs, (len(models), len(recos)))[:, i], np.reshape(std_psnr_corpdfs, (len(models), len(recos)))[:, i]))]))
    print(' & ' + ' & '.join([(r'\bfseries ' if j == jalal_argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_psnr_corpdfs, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_corpdfs, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(f' & & & {tv_ssim_corpdfs[i][0]:.2f} \pm {tv_ssim_corpdfs[i][1]:.2f} & {unet_ssim_corpdfs[i][0]:.2f} \pm {unet_ssim_corpdfs[i][1]:.2f} &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(mean_ssim_corpdfs, (len(models), len(recos)))[:, i], np.reshape(std_ssim_corpdfs, (len(models), len(recos)))[:, i]))]))
    print(' &' + ' & '.join([(r'\bfseries ' if j == jalal_argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_ssim_corpdfs, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_corpdfs, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\mymidrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{Brain}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(mean_psnr_brain, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(mean_ssim_brain, (len(models), len(recos)))[:, i])
    jalal_argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_brain, (len(models), len(recos)))[:, i])
    jalal_argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_brain, (len(models), len(recos)))[:, i])
    print(f' & {tv_psnr_brain[i][0]:.2f} \pm {tv_psnr_brain[i][1]:.2f} & {unet_psnr_brain[i][0]:.2f} \pm {unet_psnr_brain[i][1]:.2f} & ' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(mean_psnr_brain, (len(models), len(recos)))[:, i], np.reshape(std_psnr_brain, (len(models), len(recos)))[:, i]))]))
    print(' & ' + ' & '.join([(r'\bfseries ' if j == jalal_argmax_psnr else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_psnr_brain, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_brain, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(f' & & & {tv_ssim_brain[i][0]:.2f} \pm {tv_ssim_brain[i][1]:.2f} & {unet_ssim_brain[i][0]:.2f} \pm {unet_ssim_brain[i][1]:.2f} &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(mean_ssim_brain, (len(models), len(recos)))[:, i], np.reshape(std_ssim_brain, (len(models), len(recos)))[:, i]))]))
    print(' &' + ' & '.join([(r'\bfseries ' if j == jalal_argmax_ssim else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std) in enumerate(zip(np.reshape(jalal_mean_ssim_brain, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_brain, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\bottomrule')
print(r'\end{tabular}')



print('\n\n-----------------------------------------\n\n')

# Tables
print(r'\sisetup{table-alignment-mode = format, table-number-alignment = center, mode = text}'
      r'\begin{tabular}{ccc*{2}{S[table-format=2.2(1),separate-uncertainty,table-align-uncertainty,detect-weight]}*{5}{S[table-format=2.2(1),separate-uncertainty,table-align-uncertainty,detect-weight]}|*{5}{S[table-format=2.2(1),separate-uncertainty,table-align-uncertainty,detect-weight]}}\toprule')
print(r' & & & & & \multicolumn{5}{c}{Chung \& Ye} & \multicolumn{5}{c}{Jalal~\etal}\\')
print(r' & & {AF} & {TV} & {U-Net}' + (' & ' + ' & '.join([r'{$d=4*$}', r'{$d=4$}', r'{$d=3$}', r'{$d=2$}', r'{$d=1$}'])) * 2 + r'\\\mymidrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{CORPD}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(mean_psnr_corpd_celeba, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(mean_ssim_corpd_celeba, (len(models), len(recos)))[:, i])
    jalal_argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_corpd_celeba, (len(models), len(recos)))[:, i])
    jalal_argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_corpd_celeba, (len(models), len(recos)))[:, i])
    print(r' & {\textemdash} & {\textemdash} & ' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(mean_psnr_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(std_psnr_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(mean_psnr_corpd, (len(models), len(recos)))[:, i]))])) #print(f' & {tv_psnr_corpd[i][0]:.2f} \pm {tv_psnr_corpd[i][1]:.2f} & {unet_psnr_corpd[i][0]:.2f} \pm {unet_psnr_corpd[i][1]:.2f} & ' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(mean_psnr_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(std_psnr_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(mean_psnr_corpd, (len(models), len(recos)))[:, i]))]))
    print(' &' + ' & '.join([(r'\bfseries ' if j == jalal_argmax_psnr else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(jalal_mean_psnr_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_mean_psnr_corpd, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(r' & & & {\textemdash} & {\textemdash} &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(mean_ssim_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(std_ssim_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(mean_ssim_corpd, (len(models), len(recos)))[:, i]))])) #print(f' & & & {tv_ssim_corpd[i][0]:.2f} \pm {tv_ssim_corpd[i][1]:.2f} & {unet_ssim_corpd[i][0]:.2f} \pm {unet_ssim_corpd[i][1]:.2f} &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(mean_ssim_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(std_ssim_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(mean_ssim_corpd, (len(models), len(recos)))[:, i]))]))
    print(' &' + ' & '.join([(r'\bfseries ' if j == jalal_argmax_ssim else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(jalal_mean_ssim_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_corpd_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_mean_ssim_corpd, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\mymidrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{CORPDFS}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(mean_psnr_corpdfs_celeba, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(mean_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i])
    jalal_argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_corpdfs_celeba, (len(models), len(recos)))[:, i])
    jalal_argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i])
    print(r' & {\textemdash} & {\textemdash} &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(mean_psnr_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(std_psnr_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(mean_psnr_corpdfs, (len(models), len(recos)))[:, i]))])) #print(f' & {tv_psnr_corpdfs[i][0]:.2f} \pm {tv_psnr_corpdfs[i][1]:.2f} & {unet_psnr_corpdfs[i][0]:.2f} \pm {unet_psnr_corpdfs[i][1]:.2f} &' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(mean_psnr_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(std_psnr_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(mean_psnr_corpdfs, (len(models), len(recos)))[:, i]))]))
    print(' &' + ' & '.join([(r'\bfseries ' if j == jalal_argmax_psnr else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(jalal_mean_psnr_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_mean_psnr_corpdfs, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(r' & & & {\textemdash} & {\textemdash} &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(mean_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(std_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(mean_ssim_corpdfs, (len(models), len(recos)))[:, i]))])) #print(f' & & & {tv_ssim_corpdfs[i][0]:.2f} \pm {tv_ssim_corpdfs[i][1]:.2f} & {unet_ssim_corpdfs[i][0]:.2f} \pm {unet_ssim_corpdfs[i][1]:.2f} &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(mean_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(std_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(mean_ssim_corpdfs, (len(models), len(recos)))[:, i]))]))
    print(' &' + ' & '.join([(r'\bfseries ' if j == jalal_argmax_ssim else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(jalal_mean_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_corpdfs_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_mean_ssim_corpdfs, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\mymidrule')
print(r'\multirow{10}{*}{\rotatebox[origin=c]{90}{Brain}}')
for i in range(len(recos)):
    #for j in range(len(models)):
    print(f'&\multirow{"{2}"}{"{*}"}{"{" + masks[i] + "}"}' + f'&\multirow{"{2}"}{"{*}"}{"{" + acc[i] + "}"}')
    argmax_psnr = np.argmax(np.reshape(mean_psnr_brain_celeba, (len(models), len(recos)))[:, i])
    argmax_ssim = np.argmax(np.reshape(mean_ssim_brain_celeba, (len(models), len(recos)))[:, i])
    jalal_argmax_psnr = np.argmax(np.reshape(jalal_mean_psnr_brain_celeba, (len(models), len(recos)))[:, i])
    jalal_argmax_ssim = np.argmax(np.reshape(jalal_mean_ssim_brain_celeba, (len(models), len(recos)))[:, i])
    print(r' & {\textemdash} & {\textemdash} & ' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(mean_psnr_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(std_psnr_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(mean_psnr_brain, (len(models), len(recos)))[:, i]))])) #print(f' & {tv_psnr_brain[i][0]:.2f} \pm {tv_psnr_brain[i][1]:.2f} & {unet_psnr_brain[i][0]:.2f} \pm {unet_psnr_brain[i][1]:.2f} & ' + ' & '.join([(r'\bfseries ' if j == argmax_psnr else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(mean_psnr_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(std_psnr_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(mean_psnr_brain, (len(models), len(recos)))[:, i]))]))
    print(' &' + ' & '.join([(r'\bfseries ' if j == jalal_argmax_psnr else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(jalal_mean_psnr_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_std_psnr_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_mean_psnr_brain, (len(models), len(recos)))[:, i]))]) + r'\\')
    print(r' & & & {\textemdash} & {\textemdash} &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(mean_ssim_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(std_ssim_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(mean_ssim_brain, (len(models), len(recos)))[:, i]))])) #print(f' & & & {tv_ssim_brain[i][0]:.2f} \pm {tv_ssim_brain[i][1]:.2f} & {unet_ssim_brain[i][0]:.2f} \pm {unet_ssim_brain[i][1]:.2f} &' + ' & '.join([(r'\bfseries ' if j == argmax_ssim else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(mean_ssim_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(std_ssim_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(mean_ssim_brain, (len(models), len(recos)))[:, i]))]))
    print(' &' + ' & '.join([(r'\bfseries ' if j == jalal_argmax_ssim else '') + (r'\itshape ' if mean > mean_f else '') + f'{mean:.2f} \pm {std:.2f}' for j, (mean, std, mean_f) in enumerate(zip(np.reshape(jalal_mean_ssim_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_std_ssim_brain_celeba, (len(models), len(recos)))[:, i], np.reshape(jalal_mean_ssim_brain, (len(models), len(recos)))[:, i]))]) + r'\\')
print(r'\bottomrule')
print(r'\end{tabular}')