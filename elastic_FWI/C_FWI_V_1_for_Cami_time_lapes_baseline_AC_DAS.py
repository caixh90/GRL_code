import os
import numpy as np
import torch
torch.manual_seed(3589)
import torch.utils.data
import torch.nn as nn
from scipy.io import loadmat
from rnn_fd_elastic2_1D_kernel_DAS import rnn2D
import time
from Normalization import Nromalization_records_min_max
from RNN_FWI_objective_function import FWI_costfunction
import math
import matplotlib.pyplot as plt

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return [repackage_hidden(v) for v in h]

#############################################################################################
# ##                    Full Waveform Inversion  Model                                    ###
#############################################################################################
class FWI2D():
    def __init__(self, 
                 segment_size,
                 vmodel1, 
                 vmodel2,
                 vmodel3,
                 vmodel1_bs,
                 vmodel2_bs,
                 vmodel3_bs,
                 lambda1,
                 lambda2,
                 lambda3,
                 mute,
                 mute_type, # 1 for directwave, 2 for reflective wave
                 mask_grad,
                 total_variation_decay,
                 vp_hor_decay,               
                 wavelet,
                 shots_obs_x,
                 shots_obs_z,
                 shots_obs_DAS,
                 batchsize, 
                 obj_option,
                 ns, 
                 nz,
                 nx,
                 zs,
                 xs,
                 zr, 
                 xr,
                 dz,
                 dt,
                 nt,
                 npad, 
                 order, 
                 vmax,
                 vpadding,
                 freeSurface,
                 dtype,
                 device
                 ):
        """
        Args:
            segment_size            (int)        ---- the total discrete  number of t
            mean1,                  (float32)    ---- vp mean obtained from well log                         
            std1,                   (float32)    ---- vp std obtained from well log     
            mean2,                  (float32)    ---- vs mean obtained from well log
            std2,                   (float32)    ---- vs std obtained from well log
            mean3,                  (float32)    ---- rho mean obtained from well log
            std3,                   (float32)    ---- rho std obtained from well log
            std_scale,              (float32)    ---- scaling the std in case of the well log does not well represent the target area
            neuron,                 (list of int)---- defining the number of neurons of each NN later
            omega_0,                (float32)    ---- influencing the initialization of the weights        
            activation,             (string)     ---- defining the activation functions of the neural network
            bias,                   (Boolean)    ---- if bias is included in the neural network
            outermost_linear,       (Boolean)    ---- if True, then not activation function is applied on the last layer.
            ns,                     (float32)    ---- total number of the shots
            nz,                     (float32)    ---- total number of the grid points in z direction 
            nx,                     (float32)    ---- total number of the grid points in x direction 
            zs,                     (float32)    ---- shots positions in the z direction on the computational grid
            xs,                     (float32)    ---- shots positions in the x direction on the computational grid
            zr,                     (float32)    ---- receiver positions in the z direction on the computational grid
            xr,                     (float32)    ---- receiver positions in the x direction on the computational grid
            dz,                     (float32)    ---- receiver positions in the z direction on the computational grid
            dt,                     (float32)    ---- receiver positions in the x direction on the computational grid
            nt,                     (float32)    ---- the total discrete  number of t
            npad,                   (int)        ---- number of the PML absorbing layers
            order,                  (int)        ---- order of spatial finite difference
            vmax,                   (float32)    ---- the maximum value of the vs velocity for stable condiction calculation 
            vpadding,               (Boolean)    ---- if the elastic model needs to be padded
            freeSurface,            (Boolean)    ---- if we need the freeSurface modeling condition
            dtype,                  (dtype)      ---- the default datatype
            device,                 (device)     ---- CPU or GPU
        """
        super(FWI2D, self).__init__()
        self.device= device
        self.vmodel1 = vmodel1.requires_grad_(True).to(self.device)
        self.vmodel2 = vmodel2.requires_grad_(True).to(self.device)
        self.vmodel3 = vmodel3.requires_grad_(True).to(self.device)
        # Baseline models don't need gradients - detach to improve efficiency
        self.vmodel1_bs = vmodel1_bs.detach().to(self.device).requires_grad_(False)
        self.vmodel2_bs = vmodel2_bs.detach().to(self.device).requires_grad_(False)
        self.vmodel3_bs = vmodel3_bs.detach().to(self.device).requires_grad_(False)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.mute = mute
        self.mute_type = mute_type
        self.mask_grad= mask_grad
        self.nz = nz
        self.nx = nx
        self.ns=ns
        self.zs = zs
        self.xs = xs
        self.zr=zr 
        self.xr=xr
        self.dz=dz
        self.dt=dt
        self.nt = nt
        self.npad = npad
        self.order = order
        self.vmax=vmax
        self.vpadding=vpadding
        self.freeSurface=freeSurface
        self.dtype=dtype
        self.segment_size = segment_size
        self.total_variation_decay = total_variation_decay
        self.vp_hor_decay = vp_hor_decay
            
        self.obj_option = obj_option
        self.batchsize = batchsize
        self.wavelet = wavelet
       

        # setting up the coordinarte 
        self.nx_pad = self.nx + 2 * self.npad
        self.nz_pad = self.nz + self.npad if self.freeSurface else self.nz + 2 * self.npad
        self.N = self.nz*self.nx

        self.g_k_v = torch.zeros(self.N*3,1)
        self.d_k_v = torch.zeros(self.N*3,1)
        
        
        self.t = self.dt * torch.arange(0, self.nt, dtype=self.dtype)                 # create time vector
        
        #self.rnn = rnn2D( self.nz, self.nx, self.zs, self.xs, self.zr, self.xr, self.dz, self.dt, self.npad, self.order, self.vmax, self.freeSurface, self.dtype, self.device).to(self.device)

        self.Nromalization_records_min_max_func  = Nromalization_records_min_max(normalization_method = 2, dtype = torch.float32, device = device)
        self.train_loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(\
                                                         torch.as_tensor(self.xs),\
                                                         torch.as_tensor(self.zs),\
                                                         shots_obs_x.reshape(len(self.xs), 1, self.nt, len(self.xr)),\
                                                         shots_obs_z.reshape(len(self.xs), 1, self.nt, len(self.xr)),\
                                                         shots_obs_DAS.reshape(len(self.xs), 1, self.nt, len(self.xr)),\
                                                         self.mute),\
                                                         batch_size = self.batchsize, shuffle=False)

        self.Get_EFWI_objective_function_fn = FWI_costfunction(self.obj_option)
        
    def train(self, 
              Cut_fre_iter,
              lr,
              option=0, 
              log_interval=1, 
              results_dir=None):

        #params = self.vel_net.parameters() 
        # defining the network weights as the parameters
          
        resume_from_epoch = 0
        train_loss_history = []
        
        
        
        #defining the list which record the training history 
        vmodel1_list_baseline = [] 
        vmodel2_list_baseline = []
        vmodel3_list_baseline = []

        segment_ytPred_x_list_baseline = []
        segment_ytPred_z_list_baseline = []
        segment_ytPred_DAS_list_baseline = []


        velocity_output1_grad_list = []
        velocity_output2_grad_list = []
        velocity_output3_grad_list = []


        Max_epoch = len(Cut_fre_iter)

        coefile = loadmat("./vpvsrho_fit_coeff.mat")
        coeff = torch.as_tensor(coefile['coeff'], dtype=torch.float32).squeeze(0)
        print("this is the coeff", coeff.shape, coeff)
        for epoch in range(Max_epoch):  
            optimizer1 = torch.optim.Adam([ self.vmodel1], lr=lr[0])
            optimizer2 = torch.optim.Adam([ self.vmodel2], lr=lr[1])
            optimizer3 = torch.optim.Adam([ self.vmodel3], lr=lr[2])
            g_k_v_list = []
            d_k_v_list = []

            for iteration in range(resume_from_epoch, Cut_fre_iter[epoch][1]):
                self.g_k_v = torch.zeros(self.N*3,1)
                self.d_k_v = torch.zeros(self.N*3,1)
                t1 = time.time()
                loss, vmodel1, vmodel2, vmodel3, \
                segment_ytPred_x_normed, segment_ytPred_z_normed, segment_ytPred_DAS_normed, data_loss_baseline, \
                vmodel1_grad_accumulation, vmodel2_grad_accumulation, vmodel3_grad_accumulation = self.train_one_epoch(coeff,optimizer1, optimizer2, optimizer3, \
                                                                                                                       self.vmodel1, self.vmodel2, self.vmodel3, \
                                                                                                                       self.wavelet, option)
                t2 = time.time()
                time_used = t2-t1

                print("Epoch: {:5d}, Total Loss: {:.4e},Time: {:.4e}".format(iteration, loss.item(),time_used))
                # print("this is min value of the vp, and vs, and rho", vmodel1.min(), vmodel2.min(), vmodel3.min())

                assert (vmodel1 > 0).any()
                assert (vmodel2 > 0).any()
                assert (vmodel3 > 0).any()

                vmodel1_list_baseline.append(vmodel1.squeeze().cpu().detach().numpy())
                vmodel2_list_baseline.append(vmodel2.squeeze().cpu().detach().numpy())
                vmodel3_list_baseline.append(vmodel3.squeeze().cpu().detach().numpy())
                segment_ytPred_x_list_baseline.append(segment_ytPred_x_normed.squeeze().cpu().detach().numpy())
                segment_ytPred_z_list_baseline.append(segment_ytPred_z_normed.squeeze().cpu().detach().numpy())
                segment_ytPred_DAS_list_baseline.append(segment_ytPred_DAS_normed.squeeze().cpu().detach().numpy())


                train_loss_history.append(data_loss_baseline.cpu().detach().numpy())


                print("============== Saving files ===============")
                torch.save(train_loss_history,                    os.path.join(results_dir, 'train_loss_history.pt'))
                torch.save(vmodel1_list_baseline,                 os.path.join(results_dir,'vmodel1_list.pt'))
                torch.save(vmodel2_list_baseline,                 os.path.join(results_dir,'vmodel2_list.pt'))
                torch.save(vmodel3_list_baseline,                 os.path.join(results_dir,'vmodel3_list.pt'))
                torch.save(segment_ytPred_x_list_baseline[0],     os.path.join(results_dir,'segment_ytPred_x_normed_list_first.pt'))
                torch.save(segment_ytPred_z_list_baseline[0],     os.path.join(results_dir,'segment_ytPred_z_normed_list_first.pt'))
                torch.save(segment_ytPred_x_list_baseline[-1],    os.path.join(results_dir,'segment_ytPred_x_normed_list_final.pt'))
                torch.save(segment_ytPred_z_list_baseline[-1],    os.path.join(results_dir,'segment_ytPred_z_normed_list_final.pt'))
                torch.save(segment_ytPred_DAS_list_baseline[0],   os.path.join(results_dir,'segment_ytPred_DAS_list_first.pt'))
                torch.save(segment_ytPred_DAS_list_baseline[-1],  os.path.join(results_dir,'segment_ytPred_DAS_list_final.pt'))

                


                velocity_output1_grad_list.append(vmodel1_grad_accumulation)
                velocity_output2_grad_list.append(vmodel2_grad_accumulation)
                velocity_output3_grad_list.append(vmodel3_grad_accumulation)
                self.g_k_v[0       :self.N,   0] =  torch.as_tensor((vmodel1_grad_accumulation.squeeze().cpu().detach()).reshape(self.N), dtype=torch.float32)
                self.g_k_v[self.N  :self.N*2, 0] =  torch.as_tensor((vmodel2_grad_accumulation.squeeze().cpu().detach()).reshape(self.N), dtype=torch.float32)
                self.g_k_v[self.N*2:self.N*3, 0] =  torch.as_tensor((vmodel3_grad_accumulation.squeeze().cpu().detach()).reshape(self.N), dtype=torch.float32)
                self.d_k_v[0       :self.N,   0] =  torch.as_tensor((vmodel1.squeeze().cpu().detach()).reshape(self.N), dtype=torch.float32)
                self.d_k_v[self.N  :self.N*2, 0] =  torch.as_tensor((vmodel2.squeeze().cpu().detach()).reshape(self.N), dtype=torch.float32)
                self.d_k_v[self.N*2:self.N*3, 0] =  torch.as_tensor((vmodel3.squeeze().cpu().detach()).reshape(self.N), dtype=torch.float32)
                g_k_v_list.append(self.g_k_v.reshape(self.N*3))
                d_k_v_list.append(self.g_k_v.reshape(self.N*3))
                #print(F)
                #print(d_k_v_tensor)

                #print("Epoch: {:5d}, Loss: {:.4e}".format(epoch, loss.item()))
                
                torch.save(velocity_output1_grad_list,  os.path.join(results_dir,'velocity_output1_grad_list.pt'))
                torch.save(velocity_output2_grad_list,  os.path.join(results_dir,'velocity_output2_grad_list.pt'))
                torch.save(velocity_output3_grad_list,  os.path.join(results_dir,'velocity_output3_grad_list.pt'))
                torch.save(g_k_v_list,                  os.path.join(results_dir,'g_k_v_list.pt'))
                torch.save(d_k_v_list,                  os.path.join(results_dir,'d_k_v_list.pt'))
                

                
        return train_loss_history

    def train_one_epoch(self, coeff,optimizer1, optimizer2, optimizer3,  vmodel1, vmodel2, vmodel3, wavelet=None, option=0):
        '''
        (1) Obtaining the elastic parameters, and the synthetic data
        (2) Calculate objective function 
        (3) backward propagating the residual and update the weights in NN
        '''
        loss = 0
        vmodel1_grad_accumulation = torch.zeros_like(vmodel1.squeeze(), device=self.device)
        vmodel2_grad_accumulation = torch.zeros_like(vmodel2.squeeze(), device=self.device)
        vmodel3_grad_accumulation = torch.zeros_like(vmodel3.squeeze(), device=self.device)
        time_idx = torch.arange(self.nt, device=self.device, dtype=self.dtype).view(1, self.nt, 1)  # shape: (1, nt, 1)
        
        for batch_xs, batch_zs, shot_x, shot_z, shot_DAS, batch_mute in self.train_loader1:
            loss_Seg = 0
            ##print("this is the batch_xs", batch_xs)
            #print("this is the batch_xz", batch_zs)
            #print("this is the shape of shot_x", shot_x.shape)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            # step (1)
            vx_save, vz_save, \
            txx_save, tzz_save, txz_save, \
            segment_ytPred_x_AC,segment_ytPred_z_AC, segment_ytPred_DAS, \
            vmodel1, vmodel2, vmodel3  = self.forward_process(batch_xs, batch_zs, vmodel1, vmodel2, vmodel3, self.wavelet, option) 

            vx_save = repackage_hidden(vx_save)
            vz_save = repackage_hidden(vz_save)
            txx_save = repackage_hidden(txx_save)
            tzz_save = repackage_hidden(tzz_save)
            txz_save = repackage_hidden(txz_save)
            
            # step (2)
            # Ensure batch_mute is on the same device as time_idx
            batch_mute = batch_mute.to(self.device)
            mask = (time_idx > batch_mute.unsqueeze(1)) if self.mute_type == 1 else (time_idx < batch_mute.unsqueeze(1))
            segment_ytPred_x_AC[mask] = 0.0
            segment_ytPred_z_AC[mask] = 0.0
            segment_ytPred_DAS[mask] = 0.0
            
            segment_ytPred_x_AC  = self.Nromalization_records_min_max_func(segment_ytPred_x_AC)
            segment_ytPred_z_AC  = self.Nromalization_records_min_max_func(segment_ytPred_z_AC)
            segment_ytPred_DAS  = self.Nromalization_records_min_max_func(segment_ytPred_DAS)
            
            vs_coeff=coeff[0] * vmodel1.pow(2)+ coeff[1] * vmodel1 + coeff[2]
            # Use MSE instead of norm() - simpler and gradient-stable
            regularization_term = (vs_coeff-vmodel2).pow(2).mean()  # the relation between vs and vp 
            tv_hor_reg = torch.abs(vmodel1[:, :, :-1] - vmodel1[:, :, 1:]).mean() # the horizontal layer constrain for vp 
            loss_Seg_z_AC =  self.Get_EFWI_objective_function_fn(self.nt, None, segment_ytPred_z_AC.unsqueeze(dim=1), shot_z)
            loss_Seg_x_AC = self.Get_EFWI_objective_function_fn(self.nt, None, segment_ytPred_x_AC.unsqueeze(dim=1), shot_x)
            
            # Compute baseline difference terms efficiently - compute diffs first, then reuse
            diff1 = vmodel1 - self.vmodel1_bs
            diff2 = vmodel2 - self.vmodel2_bs
            diff3 = vmodel3 - self.vmodel3_bs
            
            # Use MSE (mean squared error) - simpler and gradient-stable
            # No need for norm() which has unstable gradient when norm→0
            diff1_mse = diff1.pow(2).mean()
            diff2_mse = diff2.pow(2).mean()
            diff3_mse = diff3.pow(2).mean()
            
            # Compute baseline_terms (lambda can be positive, negative, or zero)
            baseline_terms = self.lambda1 * diff1_mse + self.lambda2 * diff2_mse + self.lambda3 * diff3_mse
                        
            # loss_Seg = (((loss_Seg_z_AC + 0.5*loss_Seg_x_AC))*1+ 0*loss_Seg_DAS) + self.total_variation_decay*(regularization_term) + self.vp_hor_decay*tv_hor_reg
            loss_Seg = (loss_Seg_z_AC + loss_Seg_x_AC + 
                       self.total_variation_decay * regularization_term + 
                       self.vp_hor_decay * tv_hor_reg + 
                       baseline_terms)
            
            # Reduced printing frequency for better performance - only print occasionally
            print("Baseline diffs (MSE):", diff1_mse.item(), diff2_mse.item(), diff3_mse.item())

            # Print loss components to monitor training progress
            print("Loss: data_z={:.2e}, data_x={:.2e}, reg={:.2e}(w={:.2e}), tv={:.2e}(w={:.2e}), total={:.2e}".format(
                loss_Seg_z_AC.item(), loss_Seg_x_AC.item(), 
                regularization_term.item(), self.total_variation_decay,
                tv_hor_reg.item(), self.vp_hor_decay,
                loss_Seg.item()))
            print("baseline={:.2e}(λ1={:.2e},λ2={:.2e},λ3={:.2e})".format(
                baseline_terms.item(), self.lambda1, self.lambda2, self.lambda3))
            # Removed torch.save from training loop - save only when needed for debugging
            # torch.save(segment_ytPred_z_AC,  'segment_ytPred_z_AC.pt')
            # torch.save(segment_ytPred_x_AC,  'segment_ytPred_x_AC.pt')

            data_loss =  loss_Seg
            
            # step (3)
            loss_Seg.backward()
            
            vmodel1.grad.data.mul_(self.mask_grad)
            vmodel2.grad.data.mul_(self.mask_grad)
            vmodel3.grad.data.mul_(self.mask_grad)
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_([vmodel1, vmodel2, vmodel3], max_norm=1000)
            
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            
            # Clamp vmodel values to physically reasonable ranges to prevent NaN
            with torch.no_grad():
                vmodel1.data.clamp_(500, 8000)   # vp: 500-8000 m/s
                vmodel2.data.clamp_(100, 5000)   # vs: 100-5000 m/s
                vmodel3.data.clamp_(1000, 4000)  # rho: 1000-4000 kg/m³
                        
            loss += loss_Seg.detach()
            vmodel1_grad_accumulation = vmodel1_grad_accumulation + self.vmodel1.grad.squeeze().detach()
            vmodel2_grad_accumulation = vmodel2_grad_accumulation + self.vmodel2.grad.squeeze().detach()
            vmodel3_grad_accumulation = vmodel3_grad_accumulation + self.vmodel3.grad.squeeze().detach()


        return loss.cpu().detach(), \
            vmodel1, vmodel2, vmodel3 , \
            segment_ytPred_x_AC, segment_ytPred_z_AC, segment_ytPred_DAS,  \
            data_loss, vmodel1_grad_accumulation, vmodel2_grad_accumulation, vmodel3_grad_accumulation
            
    def forward_process(self, batch_xs, batch_zs, vmodel1, vmodel2, vmodel3, wavelet=None,option=0):
        # Delete old rnn to free memory before creating new one
        if hasattr(self, 'rnn') and self.rnn is not None:
            del self.rnn
            torch.cuda.empty_cache()
        self.rnn = rnn2D( self.nz, self.nx, batch_zs, batch_xs, self.zr, self.xr, self.dz, self.dt, self.npad, self.order, self.vmax, self.freeSurface, self.dtype, self.device).to(self.device)
        vx_save, vz_save, txx_save, tzz_save, txz_save, \
        segment_ytPred_x,segment_ytPred_z, segment_ytPred_vz_z,\
        _, _, _, _ = self.rnn(vmodel1, vmodel2, vmodel3, wavelet, option)
        #print("This is the shape of segment_ytPred_x.shape",segment_ytPred_x.shape)

        #print("This is the shape of segment_ytPred_z.shape",segment_ytPred_z.shape)
        
        return vx_save, vz_save, txx_save, tzz_save, txz_save, segment_ytPred_x,segment_ytPred_z, segment_ytPred_vz_z, vmodel1, vmodel2, vmodel3 
    
