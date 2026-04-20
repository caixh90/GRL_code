from scipy.ndimage import gaussian_filter
import torch
import numpy as np
import torch.nn as nn

class H_Smooth(torch.nn.Module):
    '''
    Gaussian smoothing horizontally
    '''
    def __init__(self, feature, h_smooth):
        super(H_Smooth, self).__init__()
        self.feature = feature
        self.h_smooth = h_smooth
        self.out_put_feature  = np.zeros_like(self.feature)
        self.nz, self.nx = self.feature.squeeze().shape
    def forward(self):
        for i in range(self.nz):
            self.out_put_feature[i,:] = gaussian_filter(self.feature.squeeze()[i,:], self.h_smooth)
        return self.out_put_feature 
            