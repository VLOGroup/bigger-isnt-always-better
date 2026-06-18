import torch as th
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from pathlib import Path

from physics.ct import CT, CTNew, CTFanBeam
import mydata

_, test_dl = mydata.create_ct_dataloader(Path('/media/lukasglaszner/data/datasets/CT/thorax/'), 320, batch_size=1)

ct = CT(img_width=320, n_views=60, end_angle=180, circle=True, device='cuda:0')
ct_new = CTNew(img_width=320, n_views=60, end_angle=180, circle=True, device='cuda:0')
ct_fan = CTFanBeam(img_width=320, n_views=60, end_angle=180, circle=False, device='cuda:0')

for img in test_dl:
    img = img.cuda()
    sino = ct.A(img)
    sin_new = ct_new.A(img)
    print(sino.shape, sin_new.shape)
    print(th.norm(sino.squeeze() - sin_new.squeeze().T))
    rec = ct.AT(sino)
    rec_new = ct_new.AT(sin_new)
    print(f'max: {rec.max()}, new max: {rec_new.max()}')
    rec /= rec.max()
    rec_new /= rec_new.max()
    print(th.norm(rec - rec_new))
    plt.figure(figsize=(12,8))
    plt.subplot(3,2,1)
    plt.title('Original')
    plt.imshow(img.squeeze().cpu(), cmap='gray')
    plt.subplot(3,2,2)
    plt.title('Sinogram CT')
    plt.imshow(sino.squeeze().cpu(), cmap='gray')
    plt.subplot(3,2,3)
    plt.title('Reconstruction CT')
    plt.imshow(rec.squeeze().cpu(), cmap='gray')
    plt.subplot(3,2,4)
    plt.title('Reconstruction CT New')
    plt.imshow(rec_new.squeeze().cpu(), cmap='gray')
    plt.subplot(3,2,5)
    plt.title('Difference CT')
    plt.imshow((sino.squeeze().cpu() - sin_new.squeeze().cpu().T).abs(), cmap='gray')
    plt.subplot(3,2,6)
    plt.title('Difference Reconstruction')
    plt.imshow((rec - rec_new).squeeze().cpu().abs(), cmap='gray')
    plt.show()

    sino_fan = ct_fan.A(img)
    rec_fan = ct_fan.AT(sino_fan)
    rec_fan /= rec_fan.max()

    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.title('Reconstruction CT New')
    plt.imshow(rec_new.squeeze().cpu(), cmap='gray')
    plt.subplot(1,3,2)
    plt.title('Reconstruction CT Fan Beam')
    plt.imshow(rec_fan.squeeze().cpu(), cmap='gray')
    plt.subplot(1,3,3)
    plt.title('Difference Reconstruction')
    plt.imshow((rec_new - rec_fan).squeeze().cpu().abs(), cmap='gray')
    plt.show()
    print(sino_fan.shape)

    break
