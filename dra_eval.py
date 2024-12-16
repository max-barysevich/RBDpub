# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from env import DiscreteIllumControlEnv
from uformer import Uformer

#%%

env = DiscreteIllumControlEnv('davis_data',
                              render_mode=None,
                              img_dim=128,
                              episode_length=None,
                              max_pauses_per_seq=3,
                              pause_dur=[5,30],
                              termination_psnr=None,
                              num_seqs=7,
                              frames_per_seq=None,
                              crossfade_over=7,
                              augment_scale=.2,
                              augment_shear=45,
                              augment_brightness=.5,
                              augment_contrast=[10,100],
                              salt_prob=.002,
                              salt_sigma=[.3,.8],
                              oof_sigma=[3,10],
                              oof_brightness=[0.,0.],
                              illum_ratio=[5,6],
                              exp_t_ratio=[5,30],
                              default_ise_product=[100,101],
                              bleach_exp_fudge=[.05,.051],
                              gauss_mean_frac=[.005,.02],
                              charbonnier_epsilon=1e-3,
                              reward_bonus=.2)

denoiser = Uformer(img_dim=128,
                   img_ch=1,
                   proj_dim=32,
                   proj_patch_dim=16,
                   attn_heads=[1,2,4,8],
                   attn_dim=32,
                   dropout_rate=.1,
                   leff_filters=32,
                   n=[2,2,2,2]).to('cuda')

# try the new oof denoiser
checkpoint_denoiser = torch.load('saved_weights/run20221120T1311/epoch_0120.pth')
#checkpoint_denoiser = torch.load('saved_weights/run20221126T1210/epoch_0120.pth')
denoiser.load_state_dict(checkpoint_denoiser['model'])

#%%

psnr_dra_all = []
psnr_def_all = []

seeds = 1

for seed in range(seeds):

    env.seed(seed)
    psnr_dra = env.render_non_rl(denoiser)
    psnr_dra_all.append(psnr_dra)
    
    env.seed(seed)
    psnr_def = env.render_non_rl(denoiser,dra=False)
    psnr_def_all.append(psnr_def)

psnr_dra_all = [[psnr.item() for psnr in psnr_dra] for psnr_dra in psnr_dra_all]
psnr_def_all = [[psnr.item() for psnr in psnr_def] for psnr_def in psnr_def_all]

column_names = ['dra_'+str(seed) for seed in range(seeds)] + ['def_'+str(seed) for seed in range(seeds)]

d_dra = pd.DataFrame(
    psnr_dra_all
    ).T.set_axis(['dra_'+str(seed) for seed in range(seeds)],
                 axis=1)
d_dra.to_csv('dra_psnr_eval.csv')

d_def = pd.DataFrame(
    psnr_def_all
    ).T.set_axis(['def_'+str(seed) for seed in range(seeds)],axis=1)
d_def.to_csv('def_psnr_eval.csv')

#%%
root = 'performance/er in silico/1/'
d_dra = pd.read_csv(root+'dra_psnr_eval.csv',index_col=0)
d_def = pd.read_csv(root+'def_psnr_eval.csv',index_col=0)

lim = 200

dra_mean = np.array(d_dra.mean(axis=1).rolling(10).mean()[9:lim+9])
dra_min = d_dra.min(axis=1).rolling(10).mean()[9:lim+9]
dra_max = d_dra.max(axis=1).rolling(10).mean()[9:lim+9]
dra_std = np.array(d_dra.std(axis=1).rolling(10).mean()[9:lim+9])

plt.plot(dra_mean)
plt.fill_between(range(dra_std.shape[0]),
                 dra_mean-dra_std,
                 dra_mean+dra_std,
                 alpha=.1)

def_mean = np.array(d_def.mean(axis=1).rolling(10).mean()[9:lim+9])
def_min = d_def.min(axis=1).rolling(10).mean()[9:lim+9]
def_max = d_def.max(axis=1).rolling(10).mean()[9:lim+9]
def_std = np.array(d_def.std(axis=1).rolling(10).mean()[9:lim+9])

plt.plot(def_mean)
plt.fill_between(range(def_std.shape[0]),
                 def_mean-def_std,
                 def_mean+def_std,
                 alpha=.1)

#plt.xlabel('Frame number')
#plt.ylabel('PSNR / dB')
plt.xlim([0,lim])
#plt.ylim([0,35])
#plt.box(False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

#%%

d = pd.DataFrame({'dra_mean':dra_mean,
                  'def_mean':def_mean,
                  'def_std':def_std,
                  'dra_std':dra_std})

df = d.fillna(0)

df.to_csv('_eval.csv')
