
import numpy as np
import torch
import math
import matplotlib.pyplot as plt 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FWI_costfunction(torch.nn.Module):
    def __init__(self, obj_option):
        super(FWI_costfunction, self).__init__()
        self.obj_option = obj_option
        if self.obj_option == 1:
            print("Using the L1 Norm objective function")
        if self.obj_option == 2:
            print("Using the L2 Norm objective function")
        if self.obj_option == 3:
            print("Using the Correlation based objective function")
        if self.obj_option == 4:
            print("Using the Zero Mean Correlation based objective function")
        if self.obj_option ==5:
            print("using the multi-scale Z transform")
            
        
    
    def L2_norm(self,shot_syn, shot_obs):
        # VECTORIZED VERSION - much faster than loop-based
        # shot_syn, shot_obs shape: (num_batch, num_shot, nt, nr)
        
        # Compute sum of obs for each shot to create mask
        obs_sum = shot_obs.abs().sum(dim=(2, 3))  # (num_batch, num_shot)
        valid_mask = obs_sum > 1e-20  # shots with non-zero data
        
        # Compute MSE for all shots at once
        diff_sq = (shot_syn - shot_obs).pow(2)  # (num_batch, num_shot, nt, nr)
        shot_mse = diff_sq.sum(dim=(2, 3))  # (num_batch, num_shot)
        
        # Apply mask and sum
        loss_Seg = (shot_mse * valid_mask).sum()
        
        return loss_Seg
    
    
    def L1_norm(self,shot_syn, shot_obs):
        # VECTORIZED VERSION - much faster than loop-based
        # shot_syn, shot_obs shape: (num_batch, num_shot, nt, nr)
        
        # Compute sum of obs for each shot to create mask
        obs_sum = shot_obs.abs().sum(dim=(2, 3))  # (num_batch, num_shot)
        valid_mask = obs_sum > 1e-20  # shots with non-zero data
        
        # Compute L1 for all shots at once
        diff_abs = (shot_syn - shot_obs).abs()  # (num_batch, num_shot, nt, nr)
        shot_l1 = diff_abs.sum(dim=(2, 3))  # (num_batch, num_shot)
        
        # Apply mask and sum
        loss_Seg = (shot_l1 * valid_mask).sum()
        
        return loss_Seg
    
    
    def global_correlation_misfit(self, shot_syn, shot_obs):
        [num_batch, num_shot] = [shot_syn.shape[0],shot_syn.shape[1]]
        loss_Seg = 0
        for ibatch in range(num_batch):
            for ishot in range (num_shot):
                i_shot_true =  shot_obs[ibatch, ishot,:,:]
                i_shot_pred =  shot_syn[ibatch, ishot,:,:]

                if torch.abs(torch.sum(i_shot_true)) <=1e-20:
                    #print("this is torch.sum(i_shot_true)", torch.sum(i_shot_true))
                    loss_Seg = 0
                else:
                    correlation_shot_true_pred = i_shot_true*i_shot_pred
                    correlation_shot_true_true = i_shot_true*i_shot_true
                    correlation_shot_pred_pred = i_shot_pred*i_shot_pred
                    E_true = torch.sum(correlation_shot_true_true)
                    E_pred = torch.sum(correlation_shot_pred_pred)
                    # Add small epsilon to prevent division by zero (causes NaN)
                    eps = 1e-10
                    global_correlation_res_shot = correlation_shot_true_pred/(torch.sqrt(E_true + eps)*torch.sqrt(E_pred + eps))
                    loss_Seg = loss_Seg + torch.sum(global_correlation_res_shot)*(-1)
        return loss_Seg
    
    
    def zero_mean_global_correlation_misfit(self, shot_syn,shot_obs):
        [num_batch, num_shot] = [shot_syn.shape[0],shot_syn.shape[1]]
        loss_Seg = 0
        for ibatch in range(num_batch):
          for ishot in range (num_shot):
              i_shot_true = shot_obs[ibatch, ishot,:,:]
              i_shot_pred = shot_syn[ibatch, ishot,:,:]

              mean_shot_true_pred = torch.mean(i_shot_pred)
              mean_shot_true_true = torch.mean(i_shot_true)
              
              correlation_shot_true_pred = (i_shot_true - mean_shot_true_true)*(i_shot_pred - mean_shot_true_pred)
              correlation_shot_true_true = (i_shot_true - mean_shot_true_true)*(i_shot_true - mean_shot_true_true)
              correlation_shot_pred_pred = (i_shot_pred - mean_shot_true_pred)*(i_shot_pred - mean_shot_true_pred)
              
              E_true = torch.sum(correlation_shot_true_true)
              E_pred = torch.sum(correlation_shot_pred_pred)

              # Add small epsilon to prevent division by zero (causes NaN)
              eps = 1e-10
              global_correlation_res_shot = correlation_shot_true_pred/(torch.sqrt(E_true + eps)*torch.sqrt(E_pred + eps))
              loss_Seg = loss_Seg + torch.sum(global_correlation_res_shot)*(-1)
        
        return loss_Seg
    
    def multiscale_Z_transform(self, nt, i_ter, shot_pre_comb,shot_true_comb):
        d_t_max = nt
        self.t_lenth = nt
        self.iter = i_ter
        
        self.FFT_kernel_real = np.ones((d_t_max,d_t_max))
        self.FFT_kernel_imag = np.ones((d_t_max,d_t_max))
        if self.iter<20:
            zzzz = 1.02
            for i in range(d_t_max):
                for j in range(d_t_max):
                    self.FFT_kernel_real[i,j] = (zzzz**(-j))*np.cos((-2*np.pi*i*j)/d_t_max)
                    self.FFT_kernel_imag[i,j] = (zzzz**(-j))*np.sin((-2*np.pi*i*j)/d_t_max)
        elif 20<=self.iter<50:
            zzzz = 1.01
            for i in range(d_t_max):
                for j in range(d_t_max):
                    self.FFT_kernel_real[i,j] = (zzzz**(-j))*np.cos((-2*np.pi*i*j)/d_t_max)
                    self.FFT_kernel_imag[i,j] = (zzzz**(-j))*np.sin((-2*np.pi*i*j)/d_t_max)
        elif 50<=self.iter<70:
            zzzz = 1.005 
            for i in range(d_t_max):
                for j in range(d_t_max):
                    self.FFT_kernel_real[i,j] = (zzzz**(-j))*np.cos((-2*np.pi*i*j)/d_t_max)
                    self.FFT_kernel_imag[i,j] = (zzzz**(-j))*np.sin((-2*np.pi*i*j)/d_t_max)
        else:
            zzzz = 1 
            for i in range(d_t_max):
                for j in range(d_t_max):
                    self.FFT_kernel_real[i,j] = (zzzz**(-j))*np.cos((-2*np.pi*i*j)/d_t_max)
                    self.FFT_kernel_imag[i,j] = (zzzz**(-j))*np.sin((-2*np.pi*i*j)/d_t_max)
                #self.FFT_kernel_real[i,j] = np.cos((-2*np.pi*i*j)/d_t_max)
                #self.FFT_kernel_imag[i,j] = np.sin((-2*np.pi*i*j)/d_t_max)
        self.FFT_kernel_real = torch.as_tensor(self.FFT_kernel_real,dtype = torch.float32).to(device)
        self.FFT_kernel_imag = torch.as_tensor(self.FFT_kernel_imag,dtype = torch.float32).to(device)


        [num_batch, num_shot, num_nt, num_nr] = \
        [shot_pre_comb.shape[0],shot_pre_comb.shape[1],shot_pre_comb.shape[2],shot_pre_comb.shape[3]]

        loss_Seg = 0

        FFT_amplitude_true = torch.zeros(num_shot,num_nt,num_nr)
        FFT_phase_true     = torch.zeros(num_shot,num_nt,num_nr)

        FFT_amplitude_pred = torch.zeros(num_shot,num_nt,num_nr)
        FFT_phase_pred     = torch.zeros(num_shot,num_nt,num_nr)
        for ibatch in range(num_batch):
          for ishot in range (num_shot):
              i_shot_true =  shot_true_comb[ibatch, ishot,:,:]
              i_shot_pred =  shot_pre_comb [ibatch, ishot,:,:]
              for i_r in range(num_nr):
                  i_line_true = i_shot_true[:,i_r]
                  i_line_pred = i_shot_pred[:,i_r]
                  i_line_true = torch.reshape(torch.as_tensor(i_line_true,dtype = torch.float32),(num_nt,1))
                  #print(self.FFT_kernel_real.shape)
                  #print(i_line_true.shape)
                  true_real =  torch.mm(self.FFT_kernel_real,i_line_true)
                  true_imag =  torch.mm(self.FFT_kernel_imag,i_line_true)
                  
                  i_FFT_amplitude_true = \
                  (torch.pow(true_real,2) + torch.pow(true_imag,2)).squeeze() # give amplitude spectrum
                  i_FFT_phase_true     = \
                  (torch.arctan(true_imag/true_real)).squeeze()               # give phase     spectrum
                  
                  i_line_pred          =\
                  torch.reshape(torch.as_tensor(i_line_pred,dtype = torch.float32),(num_nt,1))
                  
                  pred_real            =  \
                  torch.mm(self.FFT_kernel_real,i_line_pred)
                  pred_imag            =  \
                  torch.mm(self.FFT_kernel_imag,i_line_pred)
                  
                  i_FFT_amplitude_true = \
                  (torch.pow(true_real,2) + torch.pow(true_imag,2)).squeeze() # give amplitude spectrum
                  i_FFT_phase_true     = \
                  (torch.atan2(true_imag,true_real)).squeeze()               # give phase     spectrum
                  
                  i_FFT_amplitude_pred = \
                  (torch.pow(pred_real,2) + torch.pow(pred_imag,2)).squeeze()
                  i_FFT_phase_pred     = \
                  (torch.atan2(pred_imag,pred_real)).squeeze()
                  
                  #log_loss = (1/2)*\
                  #((torch.log(i_FFT_amplitude_pred[:300]/i_FFT_amplitude_true[:300]))**2 + \
                  #((i_FFT_phase_pred[:300] - i_FFT_phase_true[:300])**2))
                  
                  #log_loss = (1/2)*\
                  #((torch.log(i_FFT_amplitude_pred[:300]/i_FFT_amplitude_true[:300]))**2)
                  
                  #log_loss = ((pred_imag[:250]*true_imag[:250]))/\
                  #(torch.sqrt(torch.sum(pred_imag[:250]**2))+ \
                  #            torch.sum(true_imag[:250]**2))
                  #
                  #
                  log_loss = (1/2)*((pred_real[:self.t_lenth//2] - true_real[:self.t_lenth//2])**2 + (pred_imag[:self.t_lenth//2] - true_imag[:self.t_lenth//2])**2) 
                  #log_loss = (1/2)*((i_FFT_amplitude_true[:(self.t_lenth//2)]-i_FFT_amplitude_pred[:(self.t_lenth//2)]))**2
                  
                  loss_Seg = loss_Seg + torch.sum(log_loss)
                  
              
        print("loss_Seg = ",loss_Seg)
                
        return loss_Seg
    def Envelope(self,i_ter, shot_pre_comb,shot_true_comb):
        [num_batch, num_shot, num_nt, num_nr] = [shot_pre_comb.shape[0],shot_pre_comb.shape[1],shot_pre_comb.shape[2],shot_pre_comb.shape[3]]
        loss = 0
        if (i_ter <= 30):
            for ibatch in range(num_batch):
                for ishot in range (num_shot):
                    i_shot_true =  shot_true_comb[ibatch, ishot,:,:]
                    i_shot_pred =  shot_pre_comb [ibatch, ishot,:,:]
                    for i_r in range(num_nr):
                        i_line_true = i_shot_true[:,i_r]
                        i_line_pred = i_shot_pred[:,i_r]
                        #i_line_true = torch.reshape(torch.as_tensor(i_line_true,dtype = torch.float32),(num_nt,1))
                        '''
                        i_line_pred_fft = fx = torch.fft.fft(i_line_pred)
                        i_line_pred_fft_h = i_line_pred_fft
                        i_line_pred_fft_h[1: self.t//2] = 2 * i_line_pred_fft_h[1: self.t//2]
                        i_line_pred_fft_h[self.t//2:] = 0
                        i_line_pred_fft_hilbert = torch.fft.ifft(i_line_pred_fft_h).imag

                        i_line_true_fft = fx = torch.fft.fft(i_line_true)
                        i_line_true_fft_h = i_line_true_fft
                        i_line_true_fft_h[1: self.t//2] = 2 * i_line_true_fft_h[1: self.t//2]
                        i_line_true_fft_h[self.t//2:] = 0
                        i_line_true_fft_hilbert = torch.fft.ifft(i_line_true_fft_h).imag

                        i_line_pred_envelope = torch.abs(torch.fft.ifft(i_line_pred_fft_hilbert))
                        i_line_true_envelope = torch.abs(torch.fft.ifft(i_line_true_fft_hilbert))
                        '''
                        i_line_pred_fft = torch.fft.rfft(i_line_pred)
                        i_line_pred_fft_H = -1j*i_line_pred_fft 
                        i_line_pred_fft_H_inv = torch.fft.irfft(i_line_pred_fft_H)
                        i_line_pred_envelope = torch.sqrt(i_line_pred**2 + i_line_pred_fft_H_inv**2 + 1e-30)

                        i_line_true_fft = torch.fft.rfft(i_line_true)
                        i_line_true_fft_H = -1j*i_line_true_fft
                        i_line_true_fft_H_inv = torch.fft.irfft(i_line_true_fft_H)
                        i_line_true_envelope = torch.sqrt(i_line_true**2 + i_line_true_fft_H_inv**2 + 1e-30)

                        loss = loss + torch.sum((i_line_pred_envelope - i_line_true_envelope)**2 )
        else:
             for ibatch in range(num_batch):
                for ishot in range (num_shot):
                    i_shot_true =  shot_true_comb[ibatch, ishot,:,:]
                    i_shot_pred =  shot_pre_comb [ibatch, ishot,:,:]
                    for i_r in range(num_nr):
                        i_line_true = i_shot_true[:,i_r]
                        i_line_pred = i_shot_pred[:,i_r]

                        loss = loss + torch.sum((i_line_true - i_line_pred)**2 ) 
        return loss

    import torch.nn as nn

    def exponential_phase(self, shot_pre_comb, shot_true_comb):
        [num_batch, num_shot, num_nt, num_nr] = \
        [shot_pre_comb.shape[0],shot_pre_comb.shape[1],shot_pre_comb.shape[2],shot_pre_comb.shape[3]]

        self.damper  = 1#torch.exp(-0.01*torch.arange(0, num_nt)).to(device)

        loss = 0
        for ibatch in range(num_batch):
          for ishot in range (num_shot):
              i_shot_true =  shot_true_comb[ibatch, ishot,:-2,:]
              i_shot_pred =  shot_pre_comb [ibatch, ishot,:-2,:]
              for i_r in range(num_nr):
                  i_line_true = i_shot_true[:,i_r]
                  i_line_pred = i_shot_pred[:,i_r]
                  
                  
                  #i_line_pred_hilbert = torch.as_tensor(signal.hilbert(i_line_pred.cpu())).to()
                  #i_line_true_hilbert = torch.as_tensor(signal.hilbert(i_line_true.cpu()))
                  
                  #i_line_pred_envelope = torch.sqrt(i_line_pred**2 + i_line_pred_hilbert**2)
                  #i_line_true_envelope = torch.sqrt(i_line_true**2 + i_line_true_hilbert**2)

                  i_line_pred_fft = torch.fft.rfft(i_line_pred)
                  i_line_pred_fft_H = -1j*i_line_pred_fft
                  i_line_pred_fft_H_inv = torch.fft.irfft(i_line_pred_fft_H)
                  #i_line_pred_analytic_signal = i_line_pred + 1j*i_line_pred_fft_H_inv
                  #i_line_pred_instantaneous_phase = torch.angle(i_line_pred_analytic_signal)
                  i_line_pred_envelope = torch.sqrt(i_line_pred**2 + i_line_pred_fft_H_inv**2 + 1e-20)
                  #print(i_line_pred_envelope)
                  #i_line_pred_instantaneous_phase = torch.atan2(i_line_pred_fft_H_inv, i_line_pred)


                  i_line_true_fft = torch.fft.rfft(i_line_true)
                  i_line_true_fft_H = -1j*i_line_true_fft
                  i_line_true_fft_H_inv = torch.fft.irfft(i_line_true_fft_H)
                  #i_line_true_instantaneous_phase = torch.atan2(i_line_true_fft_H_inv, i_line_true)

                  #i_line_true_analytic_signal = i_line_pred + 1j*i_line_pred_fft_H_inv
                  #i_line_true_instantaneous_phase = torch.angle(i_line_pred_analytic_signal)
                  i_line_true_envelope = torch.sqrt(i_line_pred**2 + i_line_pred_fft_H_inv**2 + 1e-20)
                  #print(i_line_true_envelope)
                  #A1  = torch.sum((i_line_pred          *i_line_true_envelope - i_line_true*i_line_pred_envelope)          *(i_line_true_envelope*i_line_pred_envelope))*(-1)
                  #A2  = torch.sum((i_line_pred_fft_H_inv*i_line_true_envelope - i_line_true_fft_H_inv*i_line_pred_envelope)*(i_line_true_envelope*i_line_pred_envelope))*(-1)
                  #A1  = torch.sum((i_line_pred/torch.sqrt((torch.sum(i_line_pred**2)))*self.damper - \
                  #                 i_line_true/torch.sqrt((torch.sum(i_line_true**2)))*self.damper)**2)
                  #A2  = torch.sum((i_line_pred_fft_H_inv/torch.sqrt(torch.sum(i_line_pred_fft_H_inv**2))*self.damper - \
                  #                 i_line_true_fft_H_inv/torch.sqrt(torch.sum(i_line_true_fft_H_inv**2))*self.damper)**2)

                  A1  = torch.sum((i_line_pred - i_line_true )**2)
                  A2  = torch.sum((i_line_pred_fft_H_inv - i_line_true_fft_H_inv)**2)
                  '''
                  if torch.isnan(A1):
                    A1 = 0
                    print("A1 is NAN")
                  if torch.isnan(A2):
                    print("A2 is NAN")
                    A2 = 0
                  '''
                  
                  loss = loss + A1 + A2
                  #loss = loss + torch.sum((i_line_pred_fft_H_inv*i_line_true_fft_H_inv - i_line_pred*i_line_true)**2)
                  #print("{:10.9f}".format(loss))
        #print("I am here")             
        return loss
    def phase_2(self, nt, shot_pre_comb,shot_true_comb):
        d_t_max = nt
        self.t_lenth = nt
        
        self.FFT_kernel_real = np.ones((d_t_max,d_t_max))
        self.FFT_kernel_imag = np.ones((d_t_max,d_t_max))

        for i in range(d_t_max):
            for j in range(d_t_max):
                self.FFT_kernel_real[i,j] = ((-j))*np.cos((-2*np.pi*i*j)/d_t_max)
                self.FFT_kernel_imag[i,j] = ((-j))*np.sin((-2*np.pi*i*j)/d_t_max)
            #self.FFT_kernel_real[i,j] = np.cos((-2*np.pi*i*j)/d_t_max)
            #self.FFT_kernel_imag[i,j] = np.sin((-2*np.pi*i*j)/d_t_max)
        self.FFT_kernel_real = torch.as_tensor(self.FFT_kernel_real,dtype = torch.float32).to(device)
        self.FFT_kernel_imag = torch.as_tensor(self.FFT_kernel_imag,dtype = torch.float32).to(device)


        [num_batch, num_shot, num_nt, num_nr] = \
        [shot_pre_comb.shape[0],shot_pre_comb.shape[1],shot_pre_comb.shape[2],shot_pre_comb.shape[3]]

        loss_Seg = 0

        FFT_amplitude_true = torch.zeros(num_shot,num_nt,num_nr)
        FFT_phase_true     = torch.zeros(num_shot,num_nt,num_nr)

        FFT_amplitude_pred = torch.zeros(num_shot,num_nt,num_nr)
        FFT_phase_pred     = torch.zeros(num_shot,num_nt,num_nr)
        for ibatch in range(num_batch):
          for ishot in range (num_shot):
              i_shot_true =  shot_true_comb[ibatch, ishot,:,:]
              i_shot_pred =  shot_pre_comb [ibatch, ishot,:,:]
              for i_r in range(num_nr):
                  i_line_true = i_shot_true[:,i_r]
                  i_line_pred = i_shot_pred[:,i_r]
                  i_line_true = torch.reshape(torch.as_tensor(i_line_true,dtype = torch.float32),(num_nt,1))
                  #print(self.FFT_kernel_real.shape)
                  #print(i_line_true.shape)
                  true_real =  torch.mm(self.FFT_kernel_real,i_line_true)
                  true_imag =  torch.mm(self.FFT_kernel_imag,i_line_true)
                  i_line_pred  = torch.reshape(torch.as_tensor(i_line_pred,dtype = torch.float32),(num_nt,1))
                  
                  pred_real =  torch.mm(self.FFT_kernel_real,i_line_pred)
                  pred_imag =  torch.mm(self.FFT_kernel_imag,i_line_pred)

                  #loss =  torch.abs((pred_real * true_imag - true_real*pred_imag)/(true_imag*pred_imag +1e-10))
                  loss =  (true_real - pred_real)**2 + (true_imag - pred_imag)**2
                  #loss = 2*torch.abs(pred_real* true_imag - true_real*pred_real) #- (0.01*torch.abs(true_imag*pred_imag))
                  loss_Seg = loss_Seg + torch.sum(loss)
                  
        print(loss_Seg)
        return loss_Seg
    
    def forward(self, nt, i_ter, shot_syn, shot_obs):
        loss_Seg = 0
        if self.obj_option == 2:
            loss_Seg = self.L2_norm(shot_syn, shot_obs)
        if self.obj_option == 1:
            loss_Seg = self.L1_norm(shot_syn, shot_obs)
        if self.obj_option == 3:
            loss_Seg = self.global_correlation_misfit(shot_syn, shot_obs)
        if self.obj_option == 4:
            loss_Seg = self.zero_mean_global_correlation_misfit(shot_syn, shot_obs)
        if self.obj_option == 6:
            loss_Seg = self.multiscale_Z_transform(nt, i_ter, shot_syn, shot_obs)
        if self.obj_option == 7:
            loss_Seg = self.Envelope(i_ter, shot_syn, shot_obs)
        if self.obj_option == 8:
            loss_Seg = self.exponential_phase(shot_syn, shot_obs)
        if self.obj_option == 9:
            loss_Seg = self.phase_2(nt, shot_syn, shot_obs)
        return loss_Seg

