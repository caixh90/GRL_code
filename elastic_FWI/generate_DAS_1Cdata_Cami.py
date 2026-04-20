import os
import torch
import torch.utils.data
import numpy as np
import torch.nn as nn
from Normalization import Nromalization_records_min_max
from scipy.io import loadmat
from scipy.signal import resample
from FWI_filter import Filter_Butter
import scipy.io as spio
from Normalization import Nromalization_records_min_max




class Reading_Cami_data_DAS(torch.nn.Module):
    def __init__(self, 
                 dt= 0.0005,
                 dr = 5,
                 cut_off_freq = 60,
                 order = 8,
                 file_location = "./",
                 device="cpu"):
        super(Reading_Cami_data_DAS, self).__init__()
        self.file_location = file_location

        Snowflake_2018_DAS_1C = spio.loadmat('./line8/Snowflake_2018_DAS_line8_revision.mat', squeeze_me=True)
        self.data_vert  = torch.as_tensor(Snowflake_2018_DAS_1C['Processed_data'], dtype=torch.float32)
        self.nshot = self.data_vert.shape[0]
        self.nt = self.data_vert.shape[1]
        self.nr = self.data_vert.shape[2]
        self.dt = dt
        self.dr = dr
        self.t = self.dt* np.arange(0, self.nt)
        self.order = order
        self.cut_off_freq = cut_off_freq
        self.device = device
        self.depth = self.nr*self.dr

        self.data_vert_4_shots = np.zeros((self.nshot, self.nt, self.nr))
        Nromalization_records_min_max_func  = Nromalization_records_min_max(normalization_method = 2, dtype = torch.float32, device = device)
        self.data_vert = Nromalization_records_min_max_func(self.data_vert.squeeze()).unsqueeze(dim=0)


        for ii in range(self.nshot):
            self.data_vert_4_shots[ii, :,:]     = self.data_vert.reshape(self.nshot, self.nt, self.nr)[ii,:,:].cpu()
        print(self.data_vert_4_shots.shape)

    def forward(self):
        Filter_Butter_fun_z = Filter_Butter(shots=torch.as_tensor(self.data_vert_4_shots, dtype= torch.float32), 
                                            butter_order=self.order, 
                                            cut_f=self.cut_off_freq, 
                                            dt=self.dt, 
                                            filter_type='lowpass',
                                            dtype=torch.float32, 
                                            device=self.device)
        data_vert_4_shots_filtered = Filter_Butter_fun_z()
        data_vert_4_shots_filtered = torch.as_tensor(data_vert_4_shots_filtered, dtype=torch.float32).to(self.device)


        return data_vert_4_shots_filtered*(-1)

