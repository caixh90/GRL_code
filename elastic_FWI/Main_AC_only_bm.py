import numpy as np
import matplotlib.pyplot as plt
from plotimagesc import imagesc
from plotimagesc import add_colorbar
import sys
print(sys.version)

import torch
import torch.nn as nn
torch.cuda.empty_cache()
print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import scipy.io as spio
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import math

from generate_source_Cami import Reading_Cami_sweep
from generate_data_Cami_AC import Reading_Cami_data_AC
from generate_DAS_1Cdata_Cami import Reading_Cami_data_DAS

dt = 0.0005
nt = 981
t = dt * np.arange(0, nt)
nr = 66
dr = 5
depth = nr*dr
# number of samples in time
print("nt=", nt)
t = dt * torch.arange(0, nt, dtype=torch.float32)    

dx =5
dz =5

nz = int(350/dz)+1
nx = int(1000/dx)+1
npad = 20


print(nz,nx)

Reading_Cami_sweep_fun = Reading_Cami_sweep(dt=0.001, # Fix this value
                                            resample_dt= 0.0005,
                                            cut_off_freq=60,
                                            order=8,
                                            nt=nt,
                                            file_location="./",
                                            device=device)
wavelet = Reading_Cami_sweep_fun()

Snowflake_line= spio.loadmat('./line4/line4_accel_V3.mat', squeeze_me=True)
data_vert  = torch.as_tensor(Snowflake_line['monitorV_ns_nt_nr'], dtype=torch.float32)
data_hmax  = torch.as_tensor(Snowflake_line['monitorH_ns_nt_nr'], dtype=torch.float32)
mute_direc = torch.as_tensor(Snowflake_line['mute_dh5'], dtype=torch.float32).T+100
mute_refl = mute_direc-20
depth = Snowflake_line['Depthdh']

Reading_Cami_data_AC_fund = Reading_Cami_data_AC(data_vert=data_vert,
                                          data_hmax=data_hmax,
                                          mute=mute_direc,
                                          mute_type=1, # 1 for directwave, 2 for reflective wave
                                          nt=nt,
                                          dt= 0.0005,
                                          dr = 5,
                                          cut_off_freq = 60,
                                          order = 8,                                          
                                          device=device)
shots_obs_xd, shots_obs_zd = Reading_Cami_data_AC_fund()

Reading_Cami_data_AC_funr = Reading_Cami_data_AC(data_vert=data_vert,
                                          data_hmax=data_hmax,
                                          mute=mute_refl,
                                          mute_type=2, # 1 for directwave, 2 for reflective wave
                                          nt=nt,
                                          dt= 0.0005,
                                          dr = 5,
                                          cut_off_freq = 60,
                                          order = 8,                                          
                                          device=device)
shots_obs_xr, shots_obs_zr = Reading_Cami_data_AC_funr()

shots_obs_DAS = torch.zeros_like(shots_obs_xd)

vmodel1_list = torch.load("./line4/results/step2_monitor_vs_bmmisfit/vmodel1_list.pt")
vmodel2_list = torch.load("./line4/results/step2_monitor_vs_bmmisfit/vmodel2_list.pt")
vmodel3_list = torch.load("./line4/results/step2_monitor_vs_bmmisfit/vmodel3_list.pt")

# Convert to torch tensor if it's a numpy array, then move to device
vp_initial = torch.as_tensor(vmodel1_list[150]).squeeze().to(device)
vs_initial = torch.as_tensor(vmodel2_list[-1]).squeeze().to(device)
rho_initial = torch.as_tensor(vmodel3_list[150]).squeeze().to(device)

vmodel1_list = torch.load("./line4/results/step2_baseline_vs_bmmisfit/vmodel1_list.pt")
vmodel2_list = torch.load("./line4/results/step2_baseline_vs_bmmisfit/vmodel2_list.pt")
vmodel3_list = torch.load("./line4/results/step2_baseline_vs_bmmisfit/vmodel3_list.pt")

# Convert to torch tensor if it's a numpy array, then move to device
vp_bs = torch.as_tensor(vmodel1_list[-1]).squeeze().to(device)
vs_bs = torch.as_tensor(vmodel2_list[-1]).squeeze().to(device)
rho_bs = torch.as_tensor(vmodel3_list[-1]).squeeze().to(device)


# 对于 numpy 数组
print("Vp:", vp_bs.min(), vp_bs.max())
print("Vs:", vs_bs.min(), vs_bs.max())
print("Rho:", rho_bs.min(), rho_bs.max())

# 或者一行
print(f"Vp: {vp_initial.min():.1f} - {vp_initial.max():.1f}, Vs: {vs_initial.min():.1f} - {vs_initial.max():.1f}, Rho: {rho_initial.min():.1f} - {rho_initial.max():.1f}")

shot_offset_array=np.array(Snowflake_line['offset'])
xs = ((500 + shot_offset_array) / dx).astype(int)
zs = np.ones(xs.shape, dtype=np.int32) # source z-coordinate
zr = (depth/dx).astype(int) # receiver x-coordinate
xr = np.ones(zr.shape, dtype=np.int32)*int(nx//2) # receiver z-coordinate   
print(shot_offset_array,xs,zr,xr)

print(shots_obs_xd.shape,shots_obs_zd.shape,shots_obs_xr.shape,shots_obs_zr.shape)

# picks=[0,10,20,30,40,50,60,70,80]
# shots_obs_xd=shots_obs_xd[picks, :, :]
# shots_obs_zd=shots_obs_zd[picks, :, :]
# shots_obs_xr=shots_obs_xr[picks, :, :]
# shots_obs_zr=shots_obs_zr[picks, :, :]

# xs=xs[picks]
# zs=zs[picks]
# mute_direc = mute_direc[picks, :]
# mute_refl = mute_refl[picks, :]
# print("xs=", xs*dx-500)

mask_pos = torch.zeros_like(vp_initial).reshape(1, nz, nx);    mask_pos[:, :, nx//2-2:] = 1
mask_neg = torch.zeros_like(mask_pos);    mask_neg[:, :, :nx//2+2] = 1
mask_all = torch.ones_like(mask_pos)


from C_FWI_V_1_for_Cami_time_lapes_baseline_AC_DAS import FWI2D
# for vp foucs on null shuttle
model_baseline = FWI2D ( segment_size=len(t),
                         vmodel1 = vp_initial.reshape(1, nz, nx),
                         vmodel2 = vs_initial.reshape(1, nz, nx),
                         vmodel3 = rho_initial.reshape(1, nz, nx),
                         vmodel1_bs = vp_bs.reshape(1, nz, nx),
                         vmodel2_bs = vs_bs.reshape(1, nz, nx),
                         vmodel3_bs = rho_bs.reshape(1, nz, nx),
                         lambda1 = 3,#0.3
                         lambda2 = 0,#half of lambda1
                         lambda3 = 0,#double of lambda1
                         mute=mute_refl,
                         mute_type=1, # 1 for directwave, 2 for reflective wave
                         mask_grad=mask_all,
                         total_variation_decay = 0,
                         vp_hor_decay = 0,
                         wavelet = wavelet,
                         shots_obs_x = shots_obs_xr,
                         shots_obs_z = shots_obs_zr,
                         shots_obs_DAS = torch.zeros_like(shots_obs_xr),
                         batchsize = 9,
                         obj_option = 2,
                         ns = shots_obs_xr.shape[0],
                         nz=nz,
                         nx=nx,
                         zs=zs,
                         xs=xs,
                         zr=zr,
                         xr=xr,
                         dz=dz,
                         dt=dt,
                         nt=nt,
                         npad=npad,
                         order=10,
                         vmax=4700,
                         vpadding=None,
                         freeSurface=True,
                         dtype=torch.float32,
                         device=device)
fre_iter_list  = [[60, 1]]
# lr_list = [4, 4, 1]
lr_list = [1,0, 0]
train_loss_history = model_baseline.train(Cut_fre_iter=fre_iter_list, \
                                 lr=lr_list,\
                                 option=0, \
                                 log_interval=1,\
                                 results_dir='./line4/results/')

# model_baseline = FWI2D ( segment_size=len(t),
#                          vmodel1 = vp_initial.reshape(1, nz, nx),
#                          vmodel2 = vs_initial.reshape(1, nz, nx),
#                          vmodel3 = rho_initial.reshape(1, nz, nx),
#                          vmodel1_bs = vp_bs.reshape(1, nz, nx),
#                          vmodel2_bs = vs_bs.reshape(1, nz, nx),
#                          vmodel3_bs = rho_bs.reshape(1, nz, nx),
#                          lambda1 = 0.5,#0.3
#                          lambda2 = 2,#half of lambda1
#                          lambda3 = 1,#double of lambda1
#                          mute=mute_refl,
#                          mute_type=1, # 1 for directwave, 2 for reflective wave
#                          mask_grad=mask_all,
#                          total_variation_decay = 0,
#                          vp_hor_decay = 0,
#                          wavelet = wavelet,
#                          shots_obs_x = shots_obs_xr,
#                          shots_obs_z = shots_obs_zr,
#                          shots_obs_DAS = torch.zeros_like(shots_obs_xr),
#                          batchsize = 9,
#                          obj_option = 2,
#                          ns = shots_obs_xr.shape[0],
#                          nz=nz,
#                          nx=nx,
#                          zs=zs,
#                          xs=xs,
#                          zr=zr,
#                          xr=xr,
#                          dz=dz,
#                          dt=dt,
#                          nt=nt,
#                          npad=npad,
#                          order=10,
#                          vmax=4700,
#                          vpadding=None,
#                          freeSurface=True,
#                          dtype=torch.float32,
#                          device=device)
# fre_iter_list  = [[60, 300]]
# # lr_list = [4, 4, 1]
# lr_list = [1,0, 1/2]
# train_loss_history = model_baseline.train(Cut_fre_iter=fre_iter_list, \
#                                  lr=lr_list,\
#                                  option=0, \
#                                  log_interval=1,\
#                                  results_dir='./line4/results2/')