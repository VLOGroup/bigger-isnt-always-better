import os
import ismrmrd
import ismrmrd.xsd
import numpy as np
import matplotlib.pyplot as plt

#from ismrmrdtools import show, transform

import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn

# def transform_kspace_to_image(k, dim=None, img_shape=None):
#     """ Computes the Fourier transform from k-space to image space
#     along a given or all dimensions

#     :param k: k-space data
#     :param dim: vector of dimensions to transform
#     :param img_shape: desired shape of output image
#     :returns: data in image space (along transformed dimensions)
#     """
#     if not dim:
#         dim = range(k.ndim)

#     img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
#     img *= np.sqrt(np.prod(np.take(img.shape, dim)))
#     return img


# def transform_image_to_kspace(img, dim=None, k_shape=None):
#     """ Computes the Fourier transform from image space to k-space space
#     along a given or all dimensions

#     :param img: image space data
#     :param dim: vector of dimensions to transform
#     :param k_shape: desired shape of output k-space data
#     :returns: data in k-space (along transformed dimensions)
#     """
#     if not dim:
#         dim = range(img.ndim)

#     k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
#     k /= np.sqrt(np.prod(np.take(img.shape, dim)))
#     return k


# # Load file
# filename = '/srv/local/lg/vol/v1.h5'
# if not os.path.isfile(filename):
#     print("%s is not a valid file" % filename)
#     raise SystemExit
# dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)

# header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
# enc = header.encoding[0]

# # Matrix size
# eNx = enc.encodedSpace.matrixSize.x
# eNy = enc.encodedSpace.matrixSize.y
# eNz = enc.encodedSpace.matrixSize.z
# rNx = enc.reconSpace.matrixSize.x
# rNy = enc.reconSpace.matrixSize.y
# rNz = enc.reconSpace.matrixSize.z

# # Field of View
# eFOVx = enc.encodedSpace.fieldOfView_mm.x
# eFOVy = enc.encodedSpace.fieldOfView_mm.y
# eFOVz = enc.encodedSpace.fieldOfView_mm.z
# rFOVx = enc.reconSpace.fieldOfView_mm.x
# rFOVy = enc.reconSpace.fieldOfView_mm.y
# rFOVz = enc.reconSpace.fieldOfView_mm.z

# # Number of Slices, Reps, Contrasts, etc.
# ncoils = header.acquisitionSystemInformation.receiverChannels
# if enc.encodingLimits.slice != None:
#     nslices = enc.encodingLimits.slice.maximum + 1
# else:
#     nslices = 1

# if enc.encodingLimits.repetition != None:
#     nreps = enc.encodingLimits.repetition.maximum + 1
# else:
#     nreps = 1

# if enc.encodingLimits.contrast != None:
#     ncontrasts = enc.encodingLimits.contrast.maximum + 1
# else:
#     ncontrasts = 1

# # TODO loop through the acquisitions looking for noise scans
# firstacq=0
# for acqnum in range(dset.number_of_acquisitions()):
#     acq = dset.read_acquisition(acqnum)
    
#     # TODO: Currently ignoring noise scans
#     if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
#         print("Found noise scan at acq ", acqnum)
#         continue
#     else:
#         firstacq = acqnum
#         print("Imaging acquisition starts acq ", acqnum)
#         break


# # Initialiaze a storage array
# all_data = np.zeros((nreps, ncontrasts, nslices, ncoils, eNz, eNy, eNx), dtype=np.complex64)

# # Loop through the rest of the acquisitions and stuff
# for acqnum in range(firstacq,dset.number_of_acquisitions()):
#     acq = dset.read_acquisition(acqnum)

#     # TODO: this is where we would apply noise pre-whitening

#     # Remove oversampling if needed
#     if eNx != rNx:
#         # xline = transform.transform_kspace_to_image(acq.data, [1])
#         # x0 = (eNx - rNx) / 2
#         # x1 = (eNx - rNx) / 2 + rNx
#         # xline = xline[:,x0:x1]
#         # acq.resize(rNx,acq.active_channels,acq.trajectory_dimensions)
#         # acq.center_sample = rNx/2
#         # # need to use the [:] notation here to fill the data
#         # acq.data[:] = transform.transform_image_to_kspace(xline, [1])
#         print('crap')
  
#     # Stuff into the buffer
#     rep = acq.idx.repetition
#     contrast = acq.idx.contrast
#     slice = acq.idx.slice
#     y = acq.idx.kspace_encode_step_1
#     z = acq.idx.kspace_encode_step_2
#     all_data[rep, contrast, slice, :, z, y, :] = acq.data

# all_data = np.transpose(all_data, (0, 1, 4, 3, 2, 5, 6))
# print(all_data.shape)
# im = transform_kspace_to_image(all_data[0,0,0,:,:,:,:], [2,3])
# im = np.sqrt(np.sum(np.abs(im) ** 2, 0, keepdims=True))
# im = np.transpose(im, (0, 1, 3, 2))
# print(im.shape)
# kspace = transform_image_to_kspace(im, [1,2,3])
# print(kspace.shape)
# plt.imsave('slice.png', im[0, :, :, 160].squeeze(), cmap='gray')
# np.save(filename[:-3], im)

models  = ['fastmri_knee_4_attention', 'fastmri_knee_4', 'fastmri_knee_3', 'fastmri_knee_2', 'fastmri_knee_1']
masks = ['gaussian1d/acc4', 'gaussian1d/acc8', 'gaussian2d/acc4', 'radial/acc11', 'poisson/acc15']

for mask in masks:
    for model in models:
        path = f'./{model}/Fourier_CS_3d_admm_tv/{mask}/lamb0.005/rho0.01/Brats18_CBICA_AAM_1/N1000/ssim_values.csv'
        path = f'./{model}/Fourier_CS_3d_admm_tv_fs/{mask}/lamb0.005/rho0.01/v1.npy/N1000/ssim_values.csv'

        print(rf"{np.mean(np.loadtxt(path, delimiter=','))} \pm {np.std(np.loadtxt(path, delimiter=',')):.2f} & ", end='')

    print('\n')