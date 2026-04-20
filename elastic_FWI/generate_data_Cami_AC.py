import os
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from Normalization import Nromalization_records_min_max
import scipy.io as spio
from scipy.io import loadmat
from scipy.signal import resample
from FWI_filter import Filter_Butter
from Normalization import Nromalization_records_min_max
import matplotlib.pyplot as plt


class Reading_Cami_data_AC(torch.nn.Module):
    def __init__(self, 
                 data_vert,
                 data_hmax,
                 mute,
                 mute_type, # 1 for directwave, 2 for reflective wave
                 nt,
                 dt= 0.0005,
                 dr = 5,
                 cut_off_freq = 60,
                 order = 8,                 
                 device="cpu"):
        super(Reading_Cami_data_AC, self).__init__()
        
        self.data_vert=data_vert
        self.data_hmax=data_hmax
        self.mute=mute
        self.mute_type = mute_type
        self.nt=nt
        self.nshot = self.data_hmax.shape[0]
        self.nt = self.data_hmax.shape[1]
        self.nr = self.data_hmax.shape[2]
        self.dt = dt
        self.dr = dr
        self.t = self.dt* np.arange(0, self.nt)
        self.order = order
        self.cut_off_freq = cut_off_freq
        self.device = device
        self.depth = self.nr*self.dr




        self.data_hmax_4_shots = np.zeros((self.nshot, self.nt, self.nr))
        self.data_vert_4_shots = np.zeros((self.nshot, self.nt, self.nr))
        self.Nromalization_records_min_max_func  = Nromalization_records_min_max(normalization_method = 3, dtype = torch.float32, device = device)
         
        for ii in range(self.nshot):
            self.data_hmax_4_shots[ii, :,:]     = self.data_hmax.reshape(self.nshot, self.nt, self.nr)[ii,:,:].cpu()
            self.data_vert_4_shots[ii, :,:]     = self.data_vert.reshape(self.nshot, self.nt, self.nr)[ii,:,:].cpu()
        print(self.data_hmax_4_shots.shape)
        print(self.data_vert_4_shots.shape)

        
       
        
    def forward(self):
        Filter_Butter_fun_x = Filter_Butter(shots=torch.as_tensor(self.data_hmax_4_shots, dtype= torch.float32), 
                                            butter_order=self.order, 
                                            cut_f=self.cut_off_freq, 
                                            dt=self.dt, 
                                            filter_type='lowpass',
                                            dtype=torch.float32, 
                                            device=self.device)
        data_hmax_4_shots_filtered = Filter_Butter_fun_x()
      

        Filter_Butter_fun_z = Filter_Butter(shots=torch.as_tensor(self.data_vert_4_shots, dtype= torch.float32), 
                                            butter_order=self.order, 
                                            cut_f=self.cut_off_freq, 
                                            dt=self.dt, 
                                            filter_type='lowpass',
                                            dtype=torch.float32, 
                                            device=self.device)
        data_vert_4_shots_filtered = Filter_Butter_fun_z()

        time_idx = torch.arange(self.nt).view(1, self.nt, 1)  # shape: (1, nt, 1)
        mask = (time_idx > self.mute.unsqueeze(1)) if self.mute_type == 1 else (time_idx < self.mute.unsqueeze(1))
        data_vert_4_shots_filtered[mask] = 0.0
        data_hmax_4_shots_filtered[mask] = 0.0
        
        data_vert_4_shots_filtered= self.Nromalization_records_min_max_func(data_vert_4_shots_filtered)
        data_hmax_4_shots_filtered= self.Nromalization_records_min_max_func(data_hmax_4_shots_filtered)
        

        return data_vert_4_shots_filtered, data_hmax_4_shots_filtered

