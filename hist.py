import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
import h5py
from torchvision.transforms.functional import center_crop
import mydata
import os
#from utils import normalize
import imageio


def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
    if args.complex.lower() in ('true', 't', 'yes', 'y', '1'):
        use_complex = True
    else:
        use_complex = False
    if args.fat_suppression.lower() in ('true', 't', 'yes', 'y', '1'):
        fat_suppression = True
    else:
        fat_suppression = False
    if args.brain.lower() in ('true', 't', 'yes', 'y', '1'):
        brain = True
        assert fat_suppression == False
    else:
        brain = False

    print('initaializing...')
    configs = importlib.import_module(f'configs.ve.{args.model}')
    config = configs.get_config()

    im_size = 320
    grad_size = 319
    avg = np.zeros((320, 320))

    def normalize(x):
        x -= np.min(x)
        x /= np.max(x) 
        return np.uint8(x * 255)


    _, test_dl = mydata.create_dataloader(config, fat_suppression=False)
    avg_corpd = np.zeros((im_size, im_size))
    grad_x_corpd = np.zeros((500, im_size, grad_size))
    grad_y_corpd = np.zeros((500, grad_size, im_size))
    for i, img in enumerate(test_dl):
        img = img.numpy().squeeze()
        avg_corpd += img
        grad_x_corpd[i] = img[:, 1:] - img[:, :-1]
        grad_y_corpd[i] = img[1:, :] - img[:-1, :]
    avg_corpd /= 500
    imageio.v2.imwrite('/srv/local/lg/figures/avg_corpd.png', normalize(avg_corpd[::-1, :]))

    _, test_dl = mydata.create_dataloader(config, fat_suppression=True)
    avg_corpdfs = np.zeros((im_size, im_size))
    grad_x_corpdfs = np.zeros((500, im_size, grad_size))
    grad_y_corpdfs = np.zeros((500, grad_size, im_size))
    for i, img in enumerate(test_dl):
        img = img.numpy().squeeze()
        avg_corpdfs += img
        grad_x_corpdfs[i] = img[:, 1:] - img[:, :-1]
        grad_y_corpdfs[i] = img[1:, :] - img[:-1, :]
    avg_corpdfs /= 500
    imageio.v2.imwrite('/srv/local/lg/figures/avg_corpdfs.png', normalize(avg_corpdfs[::-1, :]))

    test_dl = mydata.create_brain_dataloader()
    avg_brain = np.zeros((im_size, im_size))
    grad_x_brain = np.zeros((500, im_size, grad_size))
    grad_y_brain = np.zeros((500, grad_size, im_size))
    for i, img in enumerate(test_dl):
        img = img.numpy().squeeze()
        avg_brain += img
        grad_x_brain[i] = img[:, 1:] - img[:, :-1]
        grad_y_brain[i] = img[1:, :] - img[:-1, :]
    avg_brain /= 500
    imageio.v2.imwrite('/srv/local/lg/figures/avg_brain.png', normalize(avg_brain[::-1, :]))

    # test_dl = mydata.create_celeba_dataloader(config.data.image_size)
    # avg_celeba = np.zeros((3, 1024, 1024))
    # grad_x_celeba = np.zeros((30000, 3, 1024, 1023))
    # grad_y_celeba = np.zeros((30000, 3, 1023, 1024))
    # for i in range(30000):
    #     img = np.float64(np.transpose(imageio.imread(f'/srv/local/lg/CelebA-HQ-img/{i}.jpg'), (2, 0, 1))) / 255.
    #     avg_celeba += img
    #     img += np.random.rand(*img.shape) / 255.

    #     grad_x_celeba[i, :, :, :] = img[:, :, 1:] - img[:, :, :-1]
    #     grad_y_celeba[i] = img[:, 1:, :] - img[:, :-1, :]
    # avg_celeba /= 30000
    # imageio.v2.imwrite('/srv/local/lg/figures/avg_celeba.png', np.transpose(normalize(avg_celeba), (1, 2, 0)))
    
    _, test_dl = mydata.create_ct_dataloader(config.data.image_size, batch_size=1)
    avg_ct = np.zeros((im_size, im_size))
    grad_x_ct = np.zeros((len(test_dl), 3, im_size, grad_size))
    grad_y_ct = np.zeros((len(test_dl), 3, grad_size, im_size))
    for i, img in enumerate(test_dl):
        img = img.numpy().squeeze()
        avg_ct += img
        grad_x_ct[i] = img[:, 1:] - img[:, :-1]
        grad_y_ct[i] = img[1:, :] - img[:-1, :]
    avg_ct /= len(test_dl)
    imageio.v2.imwrite('/srv/local/lg/figures/avg_ct.png', normalize(avg_ct))


    test_dl = mydata.create_ct_head_dataloader(config.data.image_size, batch_size=1)
    avg_ct_head = np.zeros((im_size, im_size))
    grad_x_ct_head = np.zeros((len(test_dl), 3, im_size, grad_size))
    grad_y_ct_head = np.zeros((len(test_dl), 3, grad_size, im_size))
    for i, img in enumerate(test_dl):
        img = img.numpy().squeeze()
        avg_ct_head += img
        grad_x_ct_head[i] = img[:, 1:] - img[:, :-1]
        grad_y_ct_head[i] = img[1:, :] - img[:-1, :]
    avg_ct_head /= len(test_dl)
    imageio.v2.imwrite('/srv/local/lg/figures/avg_ct_head.png', normalize(avg_ct_head))


    n_bins = 501
    
    fig = plt.figure()
    axs = fig.subplots(ncols=2, nrows=1)
    labels = ['CORPD', 'CORPDFS', 'Brain', 'CelebA-HQ']
    #hist = [-np.log(np.histogram(x, bins=100)) for x in [grad_x_corpd.flatten(), grad_x_corpdfs.flatten(), grad_x_brain.flatten(), grad_x_celeba.flatten()]]
    #hist = np.zeros((4,n_bins), dtype=np.float64)
    bins = np.zeros((4,n_bins+1))
    n_min = -0.35
    n_max = 0.35
    # for i, (grad_x, grad_y) in enumerate(zip([grad_x_corpd, grad_x_corpdfs, grad_x_brain, grad_x_celeba], [grad_y_corpd, grad_y_corpdfs, grad_y_brain, grad_y_celeba])):
    for i, (grad_x, grad_y) in enumerate(zip([grad_x_ct, grad_x_ct_head], [grad_y_ct,  grad_y_ct_head])):
        hist_x, bins_x = np.histogram(grad_x, bins=n_bins, range=(n_min, n_max))
        hist_y, bins_y = np.histogram(grad_y, bins=n_bins, range=(n_min, n_max))
        # hist[1, :], bins[1, :] = np.histogram(grad_x_corpdfs, bins=n_bins, range=(n_min, n_max))
        # hist[2, :], bins[2, :] = np.histogram(grad_x_brain, bins=n_bins, range=(n_min, n_max))
        # hist[3, :], bins[3, :] = np.histogram(grad_x_celeba, bins=n_bins, range=(n_min, n_max))

        #print(np.where(((grad_x_celeba[3, :]))))

        hist_x = -np.log(hist_x)
        mask_x = np.isfinite(hist_x)
        bins_x = (bins_x[1:] + bins_x[:-1]) / 2.

        #ax0.set_yscale('log')
        #plt.yscale('log')
        hist_x = hist_x[mask_x]
        hist_x -= np.min(hist_x)
        #hist_x /= np.max(hist_x)
        axs[0].plot(bins_x[mask_x][1:-1], hist_x[1:-1])

        hist_y = -np.log(hist_y)
        mask_y = np.isfinite(hist_y)
        bins_y = (bins_y[1:] + bins_y[:-1]) / 2.

        #ax0.set_yscale('log')
        #plt.yscale('log')
        hist_y = hist_y[mask_y]
        hist_y -= np.min(hist_y)
        #hist_y /= np.max(hist_y)
        axs[1].plot(bins_y[mask_y][1:-1], hist_y[1:-1])

        np.savetxt(f'/srv/local/lg/figures/hist_y_{i+4}.csv', np.stack((bins_y[mask_y][1:-1], hist_y[1:-1]), axis=1), delimiter=',')
        np.savetxt(f'/srv/local/lg/figures/hist_x_{i+4}.csv', np.stack((bins_x[mask_x][1:-1], hist_x[1:-1]), axis=1), delimiter=',')

    #plt.xlabel(r'$\sqrt{( \nabla_x I )^2 + ( \nabla_y I )^2}$')
    #plt.ylabel('occurrences')
    # plt.legend(('CORPD', 'CORPDFS', 'Brain', 'CelebA'))
    # plt.tight_layout()
    # plt.savefig('/srv/local/lg/figures/hist.pdf')
    # plt.close()

    # imageio.help()
    


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--complex', type=str, help='Whether reconstructing complex- or real-valued data.', default='False')
    parser.add_argument('--fat_suppression', type=str, help='Whether reconstructing CORPDFS or CORPD data.', default='False')
    parser.add_argument('--brain', type=str, help='Whether reconstructing brain data.', default='False')
    parser.add_argument('--model', type=str, help='which config file to use', required=True)
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='gaussian1d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d', 'poisson', 'radial', 'outer'])
    parser.add_argument('--cutout', type=int, help='Size of the cutout if outer mask is used.', default=0)
    parser.add_argument('--acc_factor', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=4)
    parser.add_argument('--center_fraction', type=float, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=0.08)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--N', type=int, help='Number of iterations for score-POCS sampling', default=2000)
    parser.add_argument('--m', type=int, help='Number of corrector step per single predictor step.'
                                              'It is advised not to change this default value.', default=1)
    return parser


if __name__ == "__main__":
    main()