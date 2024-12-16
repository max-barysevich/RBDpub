# -*- coding: utf-8 -*-

import torch
from einops import rearrange
import numpy as np

from uformer import Uformer

from tqdm import tqdm
import os
import re
import tifffile

#%%

load_from = 'run20221120T1311'
load_epoch = 'epoch_0120.pth'

denoiser = Uformer(use_ref=True,
                   img_dim=512,
                   img_ch=1,
                   proj_dim=32,
                   proj_patch_dim=16,
                   attn_heads=[1,2,4,8],
                   attn_dim=32,
                   dropout_rate=.1,
                   leff_filters=32,
                   n=[2,2,2,2]).to('cuda')

checkpoint = torch.load('saved_weights/'+load_from+'/'+load_epoch)
denoiser.load_state_dict(checkpoint['model'])

#%%

def fix_order(file_list):
    l = [{'i':int(re.split('.tif',f)[0]),'fname':f} for f in file_list]
    l.sort(key=lambda x: x['i'])
    return [d['fname'] for d in l]

#%%

fpath = 'D:/mitochondria coloc 15 03 23/saved_15032023T1359/'

denoiser_in = []
        
for i in tqdm(range(1000)):
    
    if i == 0:
        continue
    
    with tifffile.TiffFile('D:/mitochondria_stacks/saved_15032023T1330_647.tif') as tif:
        page = tif.pages[i]
        imn = page.asarray()[np.newaxis,np.newaxis]
    
    if (i-1)%10 == 0:
        ref = imn
    
    denoiser_in.append([ref,imn])

#%%

denoiser_out = []

for x in tqdm(denoiser_in):
    with torch.no_grad():
        y = denoiser([torch.tensor(x[0],device='cuda',dtype=torch.float32),
                      torch.tensor(x[1],device='cuda',dtype=torch.float32)])
        denoiser_out.append(y)

#%%

to_save = [im[0,0].cpu().numpy() for im in denoiser_out]
to_save = [f[1][0,0] for f in denoiser_in]

to_save = np.stack(to_save,axis=0)

tifffile.imwrite('D:/mitochondria_stacks/saved_15032023T1330_647_proc.tif',to_save)

#%%

spath = 'exps_111122'
exp = 'saved_17112022T1515'
exp_path = spath + '/' + exp + '/'

denoiser_in = []
denoiser_out = []

for f in fix_order(os.listdir(exp_path)):

    stack = tifffile.imread(exp_path+f)

    for i,imn in enumerate(stack):
        imn = rearrange(np.clip(imn,0,None) / imn.max(),'... -> 1 1 ...')
        #imn = rearrange(np.clip(imn,0,None) / imn.max(),'(nh wh) (nw ww) -> (nh nw) 1 wh ww',wh=128,ww=128)
        if i == 0:
            ref = imn
        denoiser_in.append([ref,imn])


for x in tqdm(denoiser_in):
    with torch.no_grad():
        #x = [torch.tensor(x[0],device='cuda',dtype=torch.float32),torch.tensor(x[1],device='cuda',dtype=torch.float32)]
        #y = denoiser(x)[0,0].cpu().numpy()
        #y = rearrange(y,'(nh nw) c wh ww -> c (nh wh) (nw ww)',nh=4,nw=4)[0].cpu().numpy()
        y = x[1][0,0]
    denoiser_out.append(y)

to_save = np.stack(denoiser_out)
tifffile.imwrite(exp+'.tif',to_save)
