import torch
from utils import fft2, ifft2

class MRI():
    def __init__(self, mask, device='cuda:0'):
        self.mask = mask.to(device)
        self.norm_constant = self.AT(self.A(torch.ones_like(self.mask))).to(device)

    def A(self, x):
        return self.mask * fft2(x)

    def AT(self, y):
        return torch.real(ifft2(self.mask * y))

