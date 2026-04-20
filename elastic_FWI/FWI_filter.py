from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import torch

class Filter_Butter(torch.nn.Module):
    def __init__(self, shots, butter_order, cut_f, dt, filter_type,dtype, device):
        super(Filter_Butter, self).__init__()
        self.shots = shots
        print("input of butter filter shape", self.shots.shape)
        self.butter_order = butter_order
        self.cut_f = cut_f
        self.dt = dt
        self.fs = 1/self.dt
        self.dtype = dtype
        self.device = device
        self.filtered_data = torch.zeros_like(self.shots).to(self.device)
        self.wwww = (2*self.cut_f)/(1/self.dt)
        print("butter filter factor", self.wwww)
        self.filter_type = filter_type
        if self.filter_type=='highpass':
            self.b, self.a = signal.butter(self.butter_order, self.wwww, 'highpass')   #配置滤波器 8 表示滤波器的阶数
        if self.filter_type=='lowpass':
            self.b, self.a = signal.butter(self.butter_order, self.wwww, 'lowpass')   #配置滤波器 8 表示滤波器的阶数

    def forward(self,):
        num_shots, num_t, num_nx = self.shots.shape
        for i_num_shots in range(num_shots):
            for i_num_nx in range(num_nx):
                i_line = self.shots[i_num_shots,:,i_num_nx].cpu().numpy()
                filtedData = signal.filtfilt(self.b, self.a, i_line)  #data为要过滤的信号
                self.filtered_data[i_num_shots,:, i_num_nx] = torch.as_tensor(filtedData.copy()).to(self.device)
        
        return self.filtered_data





