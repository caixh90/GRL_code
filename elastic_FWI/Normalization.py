import os
import torch
import torch.utils.data
import numpy as np
import torch.nn as nn
import time


class Nromalization_records_min_max(torch.nn.Module):
    def __init__(self, 
                 normalization_method, 
                 dtype,  
                 device):
        super(Nromalization_records_min_max, self).__init__()
        '''
        if normalization_method == 0:   normalization_method ======>  x - mu/ (x.max() - x.min())
        if normalization_method == 1:   normalization_method ======>  x/ sqrt( x1^{2} + x2^{2} + x3^{2} + ..... xn^{2})
        if normalization_method == 2:   normalization_method ======>  x/ (x.max())
        if normalization_method == 3:   normalization_method ======>  x/ norm(x)
        
        VECTORIZED VERSION - Much faster than loop-based implementation!
        '''
        self.dtype = dtype
        self.device = device
        self.normalization_method = normalization_method
        self.epsilon = 1e-8  # For numerical stability
    
    def forward(self, shots):
        # shots shape: (num_shots, num_t, num_nx)
        # Normalize along dim=1 (time axis) for each trace
        
        if self.normalization_method == 0:
            # x - mean / (max - min)
            # Compute along time axis (dim=1)
            mean_val = shots.mean(dim=1, keepdim=True)  # (num_shots, 1, num_nx)
            max_val = shots.max(dim=1, keepdim=True)[0]  # (num_shots, 1, num_nx)
            min_val = shots.min(dim=1, keepdim=True)[0]  # (num_shots, 1, num_nx)
            normalize_data = (shots - mean_val) / (max_val - min_val + self.epsilon)
            
        elif self.normalization_method == 1:
            # x / sqrt(sum(x^2))
            # Compute L2 norm along time axis
            norm_val = torch.sqrt((shots ** 2).sum(dim=1, keepdim=True) + self.epsilon)
            normalize_data = shots / norm_val
            
        elif self.normalization_method == 2:
            # x / max(|x|)
            # Get absolute max along time axis
            max_abs = torch.max(shots.abs().max(dim=1, keepdim=True)[0], 
                               shots.abs().min(dim=1, keepdim=True)[0].abs())
            # Actually we need max of absolute values
            max_abs = shots.abs().max(dim=1, keepdim=True)[0]
            normalize_data = shots / (max_abs + self.epsilon)
            
        elif self.normalization_method == 3:
            # x / norm(x) - using sqrt(sum(x^2) + eps) for stable gradient
            norm_val = torch.sqrt((shots ** 2).sum(dim=1, keepdim=True) + self.epsilon)
            normalize_data = shots / norm_val
            
        else:
            # Default: no normalization
            normalize_data = shots
            
        return normalize_data
