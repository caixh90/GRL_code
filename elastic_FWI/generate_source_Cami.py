
import os 
import numpy as np
import torch
import torch.utils.data ## It’s already implemented in the library, providing you with a “data iterator.”
import torch.nn as nn ## It gives you a base class so you can “define your model” yourself.
from Normalization import Nromalization_records_min_max
from scipy.io import loadmat
from Filter_source import Filter_Butter
from scipy.signal import resample


class Reading_Cami_sweep(torch.nn.Module):
    def __init__(self, 
                 dt,
                 resample_dt,
                 cut_off_freq,
                 order,
                 nt,
                 file_location,                 
                 device):
        super(Reading_Cami_sweep, self).__init__() 
        self.file_location = file_location
        self.dt = dt
        self.resample_dt  = resample_dt
        self.Aries_1_150Hz_sweep_RS_vib_doc = loadmat(str(self.file_location)+"Org_sweep.mat")
        self.Aries_1_150Hz_sweep_RS_vib = self.Aries_1_150Hz_sweep_RS_vib_doc['Org_sweep']
        self.t = np.arange(0, self.Aries_1_150Hz_sweep_RS_vib.shape[0]*self.dt, self.dt)
        self.order = order
        self.cut_off_freq = cut_off_freq
        self.device = device
        self.nt = nt

    def autocorr(self, x):
        result = np.correlate(x, x, mode='full')
        print(result.shape)
        return result[result.size//2-1000:result.size//2+1000]
    
    def zero_to_min_phase(self, signal):
        spectrum = np.fft.fft(signal)
        magnitude_spectrum = np.abs(spectrum)
        log_spectrum = np.log(magnitude_spectrum + 1e-10)
        cepstrum = np.fft.ifft(log_spectrum).real
        cepstrum[1:len(cepstrum)//2] = 2 * cepstrum[1:len(cepstrum)//2]
        cepstrum[len(cepstrum)//2:] = 0
        min_phase_spectrum = np.exp(np.fft.fft(cepstrum))
        combined_spectrum = magnitude_spectrum * np.exp(1j * np.angle(min_phase_spectrum))
        min_phase_signal_full = np.fft.ifft(combined_spectrum).real
        min_phase_signal = min_phase_signal_full[:len(min_phase_signal_full)//2]

        return min_phase_signal
    
    def forward(self):
        Aries_1_150Hz_sweep_RS_vib_auto_corr = self.autocorr(self.Aries_1_150Hz_sweep_RS_vib.squeeze())
        Aries_1_150Hz_sweep_RS_vib_auto_corr = Aries_1_150Hz_sweep_RS_vib_auto_corr/Aries_1_150Hz_sweep_RS_vib_auto_corr.max()
        signal_dim = Aries_1_150Hz_sweep_RS_vib_auto_corr.squeeze().shape
        print("raw wavelet nt =", signal_dim[0])
        dt_ratio = self.dt/self.resample_dt
        resample_signal_dim = int(dt_ratio)*int(signal_dim[0])
        print("resample_wavelt nt= ", resample_signal_dim)

        Aries_1_150Hz_sweep_RS_vib_auto_corr_resample = resample(Aries_1_150Hz_sweep_RS_vib_auto_corr, resample_signal_dim)

        self.Filter_Butter_fun_z = Filter_Butter(org_signal=torch.as_tensor(Aries_1_150Hz_sweep_RS_vib_auto_corr_resample, dtype=torch.float32),
                                                 butter_order=self.order,
                                                 cut_f=self.cut_off_freq,
                                                 dt=self.resample_dt,
                                                 filter_type='lowpass',
                                                 dtype=torch.float32,
                                                 device="cpu")
        Aries_1_150Hz_sweep_RS_vib_auto_corr_reshape_filtered = self.Filter_Butter_fun_z()
        min_phase_signal = self.zero_to_min_phase(Aries_1_150Hz_sweep_RS_vib_auto_corr_reshape_filtered)

        wavelet_from_sweep = min_phase_signal[:self.nt]
        wavelet = torch.as_tensor(wavelet_from_sweep.copy(), dtype=torch.float32).to(self.device)
        wavelet = wavelet / wavelet.max()
        print("final wavelet nt= ", wavelet.shape)
        return wavelet 


    