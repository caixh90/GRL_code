from scipy import signal
import numpy as np
import torch

class Filter_Butter(torch.nn.Module):
    def __init__(self, org_signal, butter_order, cut_f, dt, filter_type,dtype, device):
        super(Filter_Butter, self).__init__()
        self.org_signal = org_signal
        print("input of butter filter shape = ", self.org_signal.shape)
        self.butter_order = butter_order
        self.cut_f = cut_f
        self.dt = dt
        self.fs = 1/self.dt
        self.dtype = dtype
        self.device = device
        self.filtered_signal = torch.zeros_like(self.org_signal).to(self.device)
        self.wwww = (2*self.cut_f)/(1/self.dt)
        print("butter filter factor = ", self.wwww)
        self.filter_type = filter_type
        if self.filter_type=='highpass':
            self.b, self.a = signal.butter(self.butter_order, self.wwww, 'highpass')   #配置滤波器 8 表示滤波器的阶数
        if self.filter_type=='lowpass':
            self.b, self.a = signal.butter(self.butter_order, self.wwww, 'lowpass')   #配置滤波器 8 表示滤波器的阶数

    def forward(self,):
        num_t = self.org_signal.squeeze().shape
        filtedData = signal.filtfilt(self.b, self.a, self.org_signal.squeeze())  #data为要过滤的信号
        self.filtered_data = filtedData

        return self.filtered_data