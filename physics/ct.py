import torch
import numpy as np
from .radon import Radon, IRadon

import CTorch.utils.geometry as geometry
from CTorch.projector.projector_interface import Projector

class CT():
    def __init__(self, img_width, n_views, end_angle=180, circle=False, device='cuda:0'):
        theta = np.linspace(0, end_angle, n_views, endpoint=False)
        theta_all = np.linspace(0, end_angle, end_angle, endpoint=False)

        self.radon = Radon(img_width, theta, circle).to(device)
        self.radon_all = Radon(img_width, theta_all, circle).to(device)
        self.iradon_all = IRadon(img_width, theta_all, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)
        self.radont = IRadon(img_width, theta, circle, use_filter=None).to(device)
        self.norm_constant = 2 * self.AT(self.A(torch.ones(1, 1, img_width, img_width))).to(device)

    def A(self, x):
        return self.radon(x)

    def A_all(self, x):
        return self.radon_all(x)

    def A_all_dagger(self, x):
        return self.iradon_all(x)

    def A_dagger(self, y):
        return self.iradon(y)

    def AT(self, y):
        return self.radont(y)


class CT_LA():
    """
    Limited Angle tomography
    """
    def __init__(self, img_width, radon_view, end_angle=180, uniform=True, circle=False, device='cuda:0'):
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
            theta_all = np.linspace(0, end_angle, end_angle, endpoint=False)
        else:
            theta = torch.arange(radon_view)
        self.radon = Radon(img_width, theta, circle).to(device)
        self.radon_all = Radon(img_width, theta_all, circle).to(device)
        self.iradon_all = IRadon(img_width, theta_all, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)
        self.radont = IRadon(img_width, theta, circle, use_filter=None).to(device)

    def A(self, x):
        return self.radon(x)

    def A_dagger(self, y):
        return self.iradon(y)

    def AT(self, y):
        return self.radont(y)

class CTFanBeam():
    def __init__(self, img_width, n_views, device='cuda:0'):
        self.nu = 3 * img_width
        self.nView = n_views

        self.geometry = geometry.Geom2D(
            nx=img_width,
            ny=img_width,
            dx=1.0,
            dy=1.0,
            nu=self.nu,
            nView=self.nView,
            viewAngles=np.arange(0,-2*np.pi,-2*np.pi/self.nView),
            du=1.0,
            detType='curve',
            SAD=[img_width * 1.5],
            SDD=[img_width * 3.0],
        )
        self.projector = Projector(self.geometry, 'proj', 'SF', 'forward')
        self.projector_T = Projector(self.geometry, 'proj', 'SF', 'back')
        self.norm_constant = self.AT(self.A(torch.ones(1, 1, img_width, img_width).to(device)))

        x = torch.randn(1, 1, img_width, img_width).cuda()
        y = torch.randn(1, 1, self.nView, self.nu).cuda()
        print('<Ax,y> = ', torch.sum(self.A(x)*y).item())
        print('<x,A^Ty> = ', torch.sum(x*self.AT(y)).item())

    def A(self, x):
        return self.projector(x)

    def AT(self, y):
        return self.projector_T(y)
