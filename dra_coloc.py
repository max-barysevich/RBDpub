import os
import tifffile
import numpy as np
import scipy
from tqdm import tqdm
import re
import pandas as pd

import torch
from einops import rearrange

from uformer import Uformer

#%% Bleaching evaluation by intensity

raw = 'D:/mitochondria_21_03/'
csvs = 'bleaching_21_03/'

b1x,b2x,b1y,b2y = (0,0,20,1380)
w = 512
get_488 = lambda im: im[b2x:b2x+w,b2y:b2y+w]
get_647 = lambda im: im[b1x:b1x+w,b1y:b1y+w]

def fix_order(file_list):
    file_list = [f for f in file_list if 'metadata' not in f]
    l = [{'i':int(re.split('.tif',f)[0]),'fname':f} for f in file_list]
    l.sort(key=lambda x: x['i'])
    return [d['fname'] for d in l]

for d in os.listdir(raw):

    key = re.search(r'T(.+)',d).group(1)
    if os.path.exists(csvs+f'bleach_{key}.csv'):
        continue

    data = {'q25':[],
            'q50':[],
            'q75':[]}

    for i,f in tqdm(enumerate(fix_order(os.listdir(raw+d)))):
        if i==0:
            continue
        # open image
        im = tifffile.imread(raw+d+'/'+f)
        # get 647
        im = get_647(im)
        # get median, IQR
        q25,q50,q75 = np.percentile(im.flatten(),[25,50,75])
        # record to dataframe
        data['q25'].append(q25)
        data['q50'].append(q50)
        data['q75'].append(q75)

    df = pd.DataFrame(data)
    df.to_csv(csvs+f'bleach_{key}.csv')

#%% Plot bleaching
import matplotlib.pyplot as plt

rbd = ['1327','1344','1544','1558','1700','1715','1735']
csvs = 'bleaching_21_03/'

def process_q(q):
    q = np.array(q.dropna())
    q = np.convolve(q,np.ones(10)/10,mode='valid')
    q = q/q[0]
    return q

for f in os.listdir(csvs):
    data = pd.read_csv(csvs+f)
    q25,q50,q75 = data['q25'],data['q50'],data['q75']

    q25 = process_q(q25)
    q50 = process_q(q50)
    q75 = process_q(q75)

    plt.plot(np.arange(q50.shape[0])/10,
             q50,
             label=f[:-4]+'_rbd' if any([r in f for r in rbd]) else f[:-4]+'_def',
             linewidth=1.,
             color='blue' if any([r in f for r in rbd]) else 'orange')

    plt.fill_between(np.arange(q50.shape[0])/10,
                     q25,
                     q75,
                     color='blue' if any([r in f for r in rbd]) else 'orange',
                     alpha=.1)

    plt.legend()

#%% Plot bleaching - collated medians

csvs = 'bleaching_21_03/'
rbd = ['1327','1344','1544','1558','1700','1715','1735']

df_rbd = []
df_def = []

for f in os.listdir(csvs):
    df = pd.read_csv(csvs+f)['q50']
    df.name = f[:-4]

    if any([r in f for r in rbd]):
        df_rbd.append(df)
    else:
        df_def.append(df)

df_rbd = pd.concat(df_rbd,axis=1)
df_def = pd.concat(df_def,axis=1)

df_rbd.to_csv(csvs+'bleach_rbd.csv')
df_def.to_csv(csvs+'bleach_def.csv')

#%%

d_rbd = pd.read_csv('bleaching_collated_21_03/bleach_rbd.csv',index_col=0)
d_def = pd.read_csv('bleaching_collated_21_03/bleach_def.csv',index_col=0)

roll = 10

d_rbd = d_rbd.rolling(roll).mean()
d_rbd = d_rbd.apply(lambda x:x/x.iloc[roll-1])

d_def = d_def.apply(lambda x:x/x.iloc[0])

roll = 50

rbd_mean = d_rbd.median(axis=1).rolling(roll).mean()[roll-1:6000+roll-1]
rbd_std = d_rbd.std(axis=1).rolling(roll).mean()[roll-1:6000+roll-1]

rbd_q25 = d_rbd.quantile(0.25,axis=1).rolling(roll).mean()[roll-1:6000+roll-1]
rbd_q75 = d_rbd.quantile(0.75,axis=1).rolling(roll).mean()[roll-1:6000+roll-1]

plt.plot(np.arange(rbd_mean.shape[0])/10,np.array(rbd_mean))
plt.fill_between(np.arange(rbd_mean.shape[0])/10,
                 #rbd_mean-rbd_std,
                 #rbd_mean+rbd_std,
                 np.array(rbd_q25),
                 np.array(rbd_q75),
                 alpha=.1)

def_mean = d_def.median(axis=1).rolling(roll).mean()[roll-1:6000+roll-1]
def_std = d_def.std(axis=1).rolling(roll).mean()[roll-1:6000+roll-1]

def_q25 = d_def.quantile(0.25,axis=1).rolling(roll).mean()[roll-1:6000+roll-1]
def_q75 = d_def.quantile(0.75,axis=1).rolling(roll).mean()[roll-1:6000+roll-1]

plt.plot(np.arange(def_mean.shape[0])/10,np.array(def_mean))
plt.fill_between(np.arange(def_mean.shape[0])/10,
                 #def_mean-def_std,
                 #def_mean+def_std,
                 np.array(def_q25),
                 np.array(def_q75),
                 alpha=.1)

plt.xlim([0,600])

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

#%%

raw = 'D:/mitochondria_21_03/'
save = 'D:/mitochondria_stacks_21_03/'

#raw = 'D:/mitochondria_19_10/'
#save = 'D:/mitochondria_stacks_19_10/'

#rbd = ['1118','1132','1143','1154','1221','1317'] # for 19_10
rbd = ['1327','1344','1544','1558','1700','1715','1735'] # for 21_03
# NB cells die and stop expressing GFP, which quickly bleaches (1544,1558,1715,1735)

#b1x,b2x,b1y,b2y = (111,128,148,1586) # for 19_10
b1x,b2x,b1y,b2y = (0,0,20,1380) # for 21_03
w = 512 # 384 for 19_10
#w = 384

get_488 = lambda im: im[b2x:b2x+w,b2y:b2y+w]/im[b2x:b2x+w,b2y:b2y+w].max()
get_647 = lambda im: im[b1x:b1x+w,b1y:b1y+w]/im[b1x:b1x+w,b1y:b1y+w].max()

denoiser = Uformer(use_ref=True,
                   img_dim=w,
                   img_ch=1,
                   proj_dim=32,
                   proj_patch_dim=16,
                   attn_heads=[1,2,4,8],
                   attn_dim=32,
                   dropout_rate=.1,
                   leff_filters=32,
                   n=[2,2,2,2]).to('cuda')
denoiser.load_state_dict(torch.load('saved_weights/run20221120T1311/epoch_0120.pth')['model'])

def fix_order(file_list):
    file_list = [f for f in file_list if 'metadata' not in f]
    l = [{'i':int(re.split('.tif',f)[0]),'fname':f} for f in file_list]
    l.sort(key=lambda x: x['i'])
    return [d['fname'] for d in l]

for d in os.listdir(raw):

    print(f'Analysing dir {d}')

    if not os.path.exists(save+d):
        os.mkdir(save+d)
    else:
        continue

    with (tifffile.TiffWriter(save+d+'/'+d+'_488.tif',bigtiff=True) as t488,
          tifffile.TiffWriter(save+d+'/'+d+'_647.tif',bigtiff=True) as t647,
          torch.no_grad()):

        for i,f in tqdm(enumerate(fix_order(os.listdir(raw+d)))):

            if i == 0:
                continue

            im = tifffile.imread(raw+d+'/'+f).astype(np.uint16)
            im488 = get_488(im)
            im647 = get_647(im)

            # denoise 647

            if any(exp in d for exp in rbd): # use rbd

                if (i-1)%10 == 0:
                    ref = im647

                denoiser_in = [
                    torch.tensor(rearrange(np.clip(ref,0,None),'... -> 1 1 ...'),
                                 device='cuda',
                                 dtype=torch.float32),
                    torch.tensor(rearrange(np.clip(im647,0,None),'... -> 1 1 ...'),
                                 device='cuda',
                                 dtype=torch.float32)
                ]

            else: # no rbd
                denoiser_in = [
                    torch.tensor(rearrange(np.clip(im647,0,None),'... -> 1 1 ...'),
                                 device='cuda',
                                 dtype=torch.float32),
                    torch.tensor(rearrange(np.clip(im647,0,None),'... -> 1 1 ...'),
                                 device='cuda',
                                 dtype=torch.float32)
                ]

            denoiser_out = denoiser(denoiser_in)[0,0].cpu().numpy()

            # denoise 488

            denoiser_in_488 = [
                torch.tensor(rearrange(np.clip(im488,0,None),'... -> 1 1 ...'),
                             device='cuda',
                             dtype=torch.float32),
                torch.tensor(rearrange(np.clip(im488,0,None),'... -> 1 1 ...'),
                             device='cuda',
                             dtype=torch.float32)
                ]
            denoiser_out_488 = denoiser(denoiser_in_488)[0,0].cpu().numpy()

            t647.write(denoiser_out)
            t488.write(denoiser_out_488)

#%%

stacks = 'D:/mitochondria_stacks/'
stacks_thresholded = 'D:/mitochondria_thresh/'
csvs = 'colocs_raw/'
# 647, 488
thresh = {'1118':[.3,.3],
          '1132':[.2,.2],
          '1143':[.1,.2],
          '1154':[.25,.25],
          '1204':[.2,.26],
          '1221':[.2,.3],
          '1235':[.2,.2],
          '1258':[.2,.25],
          '1317':[.3,.25],
          '1333':[.25,.35]}

def threshold(im,p):
    #t = np.percentile(im,p)
    return np.where(im>p,1,0)

def manders(im1,im2):
    return (np.sum(np.logical_and(im1,im2))-np.sum(np.logical_xor(im1,im2)))/np.sum(im1)
# metric should penalise presense in 647 but not 488

def bad_manders(im488,im647):
    return np.sum(np.logical_and(im647,~im488))/np.sum(im488)

def pearson(im488,im647):
    num = np.sum((im647-im647.mean())*(im488-im488.mean()))
    den = np.sqrt(np.sum((im647-im647.mean())**2)*np.sum((im488-im488.mean())**2))
    return num/den

for subdir in os.listdir(stacks):
    key = re.search(r'T(.+)',subdir).group(1)

    csv_name = 'coloc_'+key+'.csv'
    if csv_name in os.listdir(csvs):
        continue

    ts = thresh[key]

    imgs = os.listdir(stacks+subdir)
    im488 = [stacks+subdir+'/'+im for im in imgs if '488' in im][0]
    im647 = [stacks+subdir+'/'+im for im in imgs if '647' in im][0]

    if not os.path.exists(stacks_thresholded+subdir):
        os.mkdir(stacks_thresholded+subdir)

    im488t = [stacks_thresholded+subdir+'/'+im for im in imgs if '488' in im][0]
    im647t = [stacks_thresholded+subdir+'/'+im for im in imgs if '647' in im][0]

    c = []

    with (tifffile.TiffFile(im488) as t488,
          tifffile.TiffFile(im647) as t647,
          tifffile.TiffWriter(im488t) as t488t,
          tifffile.TiffWriter(im647t) as t647t):
        for page488, page647 in tqdm(zip(t488.pages,t647.pages)):

            im_488 = page488.asarray() >= ts[1]
            im_647 = page647.asarray() >= ts[0]
            c.append(bad_manders(im_488,im_647))

            t488t.write(im_488)
            t647t.write(im_647)

    c = np.array(c)
    colocs = pd.DataFrame(c,columns=[key])
    colocs.to_csv(csvs+csv_name,index=False)

#%% New thresholding

stacks = 'D:/mitochondria_stacks_21_03/'
stacks_thresholded = 'D:/mitochondria_thresh_21_03/'
csvs = 'colocs_raw_21_03/'
'''
stacks = 'D:/mitochondria_stacks_19_10/'
stacks_thresholded = 'D:/mitochondria_thresh_19_10/'
csvs = 'colocs_raw_19_10/'
'''
# find correct initial thresholds
thresh = { # for 21_03
    '1327':[0.18,0.33],
    '1344':[0.27,0.28],
    '1358':[0.30,0.10],
    '1417':[0.18,0.22],
    '1515':[0.20,0.12],
    '1526':[0.24,0.50],
    '1544':[0.24,0.22],
    '1558':[0.21,0.21],
    '1700':[0.19,0.44],
    '1715':[0.27,0.65],
    '1735':[0.26,0.27],
    '1756':[0.10,0.40]
    }
'''
thresh = { # for 19_10
    '1118':[.35,.35],
    '1132':[.25,.80],
    '1143':[.20,.15],
    '1154':[.25,.30],
    '1204':[.20,.20],
    '1221':[.25,.25],
    '1235':[.30,.30],
    '1258':[.35,.30],
    '1317':[.30,.30],
    '1333':[.30,.30]
    }
'''
N = 1 # how often to update threshold
X = 0.05 # if area of thresholded image is >5% away from previous, update threshold
Y = 1.0 # range for finding the optimal threshold (40% is too small)
M = 50 # number of candidate thresholds per frame

def manders(im1,im2):
    Sum = np.sum(im1*im2,dtype=np.uint64)
    Sqrt = np.sqrt(np.sum(im1,dtype=np.uint64)*np.sum(im2,dtype=np.uint64))
    return Sum/Sqrt

def update_threshold(im,t,a):
    # take the image
    # apply a set of thresholds 10% from the original in M steps
    ts = t + np.linspace(-Y*t,Y*t,M)
    a_new = []
    for ti in ts:
        a_new.append(np.sum(im>=ti))
    # fit a threshold-area curve
    #m, c, r, *_ = scipy.stats.linregress(a_new,ts)

    a_new = np.array(a_new)
    ts = np.array(ts)
    inds = np.argsort(a_new)
    a_new = a_new[inds]
    ts = ts[inds]

    a_u = np.unique(a_new)
    ts = np.array([np.mean(ts[a_new==a_i]) for a_i in a_u])

    try:
        spline = scipy.interpolate.CubicSpline(a_u,ts,extrapolate=False)
        t_new = spline(np.array([a]))[0]
    except:
        t_new = t
    r=0
    # predict correct threshold from a
    #t_new = m*a + c
    return t_new, r

for subdir in os.listdir(stacks):
    # find timestamp
    key = re.search(r'T(.+)',subdir).group(1)

    csv_name = 'coloc_'+key+'.csv'
    if csv_name in os.listdir(csvs):
        continue

    # get initial threshold (set manually)
    ts = thresh[key]

    imgs = os.listdir(stacks+subdir)
    im488 = [stacks+subdir+'/'+im for im in imgs if '488' in im][0]
    im647 = [stacks+subdir+'/'+im for im in imgs if '647' in im][0]

    if not os.path.exists(stacks_thresholded+subdir):
        os.mkdir(stacks_thresholded+subdir)

    im488t = [stacks_thresholded+subdir+'/'+im for im in imgs if '488' in im][0]
    im647t = [stacks_thresholded+subdir+'/'+im for im in imgs if '647' in im][0]

    c = []

    with (tifffile.TiffFile(im488) as t488,
          tifffile.TiffFile(im647) as t647,
          tifffile.TiffWriter(im488t) as t488t,
          tifffile.TiffWriter(im647t) as t647t):
        for i, (page488, page647) in enumerate(pbar := tqdm(zip(t488.pages,t647.pages),
                                                            desc=subdir)):

            if i == 0:
                im_488 = page488.asarray() >= ts[0]
                im_647 = page647.asarray() >= ts[1]

                # these should be ~preserved
                a488 = np.sum(im_488)
                a647 = np.sum(im_647)

            elif i%N == 0:
                # update threshold
                im_488 = page488.asarray() >= ts[0]
                im_647 = page647.asarray() >= ts[1]

                a488i = np.sum(im_488)
                a647i = np.sum(im_647)

                # set vars for residuals
                r488 = r647 = 0

                # if difference is more than X, update threshold
                if np.abs(a488i/a488 - 1) > X:
                    ts[0], r488 = update_threshold(page488.asarray(),ts[0],a488)

                if np.abs(a647i/a647 - 1) > X:
                    ts[1], r647 = update_threshold(page647.asarray(),ts[1],a647)

                pbar.set_postfix_str(f'Residuals: {r488}, {r647}')

            # threshold the image
            im_488 = page488.asarray() >= ts[0]
            im_647 = page647.asarray() >= ts[1]

            # colocalisation
            c.append(manders(im_488,im_647))

            t488t.write(im_488)
            t647t.write(im_647)

    # make csv
    c = np.array(c)
    colocs = pd.DataFrame(c,columns=[key])
    colocs.to_csv(csvs+csv_name,index=False)

#%%

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

csvs = 'colocs_raw_21_03/'
#csvs = 'colocs_raw_19_10/'
#rbd = ['1118','1132','1143','1154','1221','1317']
rbd = ['1327','1344','1544','1558','1700','1715','1735']
# 1118 - bad (647)
# 1132

# 1204 - bad (sudden change of focus around 4450)

for f in os.listdir(csvs):
    f = csvs+f
    data = pd.read_csv(f)

    column_name = data.columns[0]
    if column_name in ['1344','1358']:
        pass
        #continue
    #plt.plot(data[column_name]/data[column_name][0],
    #         label='rbd' if column_name in rbd else 'def',
    #         linewidth=.5)

    x = np.convolve(data[column_name].dropna(),np.ones(10)/10, mode='valid')
    x = x/x[0]

    plt.plot(x,
             label=f[:-4]+'_rbd' if column_name in rbd else f[:-4]+'_def',
             linewidth=1.,
             color='blue' if column_name in rbd else 'orange')

    #plt.plot([x[0],x[-1]],
    #         label=f[:-4]+'_rbd' if column_name in rbd else f[:-4]+'_def')

plt.legend()

#%% Collate csvs

csvs = 'colocs_raw_21_03/'
rbd = ['1327','1344','1544','1558','1700','1715','1735']

df_rbd = []
df_def = []

for f in os.listdir(csvs):
    df = pd.read_csv(csvs+f)

    if df.columns[0] in rbd:
        df_rbd.append(df)
    else:
        df_def.append(df)

df_rbd = pd.concat(df_rbd,axis=1)
df_def = pd.concat(df_def,axis=1)

df_rbd.to_csv(csvs+'colocs_rbd.csv')
df_def.to_csv(csvs+'colocs_def.csv')

#%% Plot everything properly

d_rbd = pd.read_csv('colocs_collated_21_03/colocs_rbd.csv',index_col=0)
d_def = pd.read_csv('colocs_collated_21_03/colocs_def.csv',index_col=0)

roll = 50

#rbd_mean = np.array(d_rbd.mean(axis=1).rolling(10).mean()[9:6009])
rbd_mean = d_rbd.median(axis=1).rolling(roll).mean()[roll-1:6000+roll-1]
rbd_std = d_rbd.std(axis=1).rolling(roll).mean()[roll-1:6000+roll-1]

rbd_q25 = d_rbd.quantile(0.25,axis=1).rolling(roll).mean()[roll-1:6000+roll-1]
rbd_q75 = d_rbd.quantile(0.75,axis=1).rolling(roll).mean()[roll-1:6000+roll-1]

plt.plot(np.arange(rbd_mean.shape[0])/10,np.array(rbd_mean))
plt.fill_between(np.arange(rbd_mean.shape[0])/10,
                 #rbd_mean-rbd_std,
                 #rbd_mean+rbd_std,
                 np.array(rbd_q25),
                 np.array(rbd_q75),
                 alpha=.1)

def_mean = d_def.median(axis=1).rolling(roll).mean()[roll-1:6000+roll-1]
def_std = d_def.std(axis=1).rolling(roll).mean()[roll-1:6000+roll-1]

def_q25 = d_def.quantile(0.25,axis=1).rolling(roll).mean()[roll-1:6000+roll-1]
def_q75 = d_def.quantile(0.75,axis=1).rolling(roll).mean()[roll-1:6000+roll-1]

plt.plot(np.arange(def_mean.shape[0])/10,np.array(def_mean))
plt.fill_between(np.arange(def_mean.shape[0])/10,
                 #def_mean-def_std,
                 #def_mean+def_std,
                 np.array(def_q25),
                 np.array(def_q75),
                 alpha=.1)

plt.xlim([0,600])

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

#%% Photobleaching

# assemble samples for visual evaluation

raw = 'D:/mitochondria_stacks_21_03/'
sampled = 'D:/mitochondria_stages_21_03/'

for d in tqdm(os.listdir(raw)):
    # open dir
    f = [raw+d+'/'+f for f in os.listdir(raw+d) if '488' in f][0]
    # open 488
    with (tifffile.TiffFile(f) as tif,
          tifffile.TiffWriter(sampled+f) as new):
        for i,page in enumerate(tif.pages):
            if i%500 == 0:
                im = page
                new.write(im)
