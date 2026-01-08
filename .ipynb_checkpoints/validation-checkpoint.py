#!/usr/bin/env python
# coding: utf-8

# In[27]:


import argparse
import json
import gc
from tqdm import tqdm
from joblib import Parallel, delayed
from glob import glob
import math
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd

import openslide
from PIL import Image
from IPython.display import display
import seaborn as sns
from skimage.measure import regionprops, find_contours
from skimage.draw import polygon2mask
from skimage.color import rgb2hed, hed2rgb
from cellpose import models, io

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchstain

import sys
sys.path.append('/home/hoyeon/crp/shest/')

from config import n_cores, generator, seed, set_seed, tile, Config, lower_micrometer, upper_micrometer
from data import Images
from model import Modeling


# In[2]:


sc.__version__, sd.__version__, torch.__version__


# # WSI file

# In[3]:


# wsi = f"/data0/HE/LUAD/TCGA-LUAD/TCGA-55-7724-01Z-00-DX1.31a194ac-62e3-4225-8e32-8c2a83dcdd10.svs"
# wsi = f"/data0/HE/LUAD/TCGA_LUAD/TCGA-55-8207-01Z-00-DX1.2dafc442-f927-4b0d-b197-cc8c5f86d0fc.svs"
# wsi = f"/data0/HE/LUAD/TCGA_LUAD/TCGA-55-8204-01Z-00-DX1.30ba69f3-53f1-41cc-826c-20dce3cfe86b.svs"
# wsi = f"/data0/HE/LUAD/TCGA_LUAD/TCGA-55-8505-01Z-00-DX1.D364C30D-BFB8-486B-A0D3-948FF8E90C3E.svs"
# wsi = f"/data0/HE/LUAD/TCGA_LUAD/TCGA-99-7458-01Z-00-DX1.10ea0b2c-c763-40d1-83a4-3d4ae957fdb0.svs"
wsi = f"/data0/HE/LUAD/TCGA_LUAD/TCGA-55-A48X-01Z-00-DX1.A46C6373-8458-4D55-88C3-4C70A05F9F47.svs"
# wsi = f"/data0/Xenium_V1/10X/Human_Lung_Cancer_Addon/Xenium_V1_Human_Lung_Cancer_Addon_FFPE_he_image.ome.tif"
# wsi = f"/data0/HE/LUAD/TCGA-LUAD/TCGA-55-8208-01A-01-TS1.53223698-3f49-460c-b3c7-6f4b5e492f5b.svs"
# wsi = f"/data0/HE/LUAD/TCGA-LUAD/TCGA-55-8208-01Z-00-DX1.6eccb7e2-16e4-4d25-9a1e-b370e016020f.svs" #file error
# wsi = f"/data0/HE/LUAD/TCGA-LUAD/TCGA-49-4487-01Z-00-DX1.3a3a0720-463c-430e-849b-e2f8991bdfa5.svs" #CUDA out of memory
# wsi = f"/data0/HE/LUAD/TCGA-LUAD/TCGA-L9-A7SV-01Z-00-DX1.153B8E2D-54CE-4747-A22E-7A6ADCA03DB5.svs" #file error
# wsi = f"/data0/HE/LUAD/TCGA-LUAD/TCGA-L9-A7SV-01A-01-TS1.E157B486-A95D-4D6F-9978-1B323B4B7065.svs" #runtime error
# wsi = f"/data0/HE/LUAD/TCGA-LUAD/TCGA-38-4627-01Z-00-DX1.fe406eb9-b38b-410d-aa1a-84ab8ac091c7.svs" #CUDA out of memory
slide = openslide.OpenSlide(wsi)
slide.get_thumbnail((1024, 1024))


# In[4]:


pixel_size = float(slide.properties['openslide.mpp-x'])
pixel_size


# In[5]:


slide.level_dimensions[0]


# In[6]:


get_ipython().run_cell_magic('time', '', 'image_raw = slide.read_region((0, 0), 0, slide.level_dimensions[0])\nimage = np.array(image_raw)[:, :, :3]\n')


# # Cell segmentation

# In[7]:


get_ipython().run_cell_magic('time', '', 'io.logger_setup()\nmodel = models.CellposeModel(gpu=True)\n')


# In[8]:


get_ipython().run_cell_magic('time', '', 'masks, flows, styles = model.eval(image)\n')


# In[9]:


lower_micrometer, upper_micrometer


# In[10]:


regions = regionprops(masks)
print(len(regions))
lower = int(math.ceil(lower_micrometer / pixel_size))
upper = int(math.ceil(upper_micrometer / pixel_size))
half = upper // 2
print(lower, upper, half)


# In[11]:


images = {}
centroids = {}
for region in tqdm(regions):
    cell_id = str(region.label)
    y, x = map(int, region.centroid)
    y_min, x_min, y_max, x_max = region.bbox
    nucleus = max(y_max - y_min, x_max - x_min)
    y_lower = y - half
    y_upper = y + half
    x_left = x - half
    x_right = x + half
    if y_lower < 0 or y_upper > image.shape[0] or x_left < 0 or x_right > image.shape[1] or nucleus < lower or nucleus > upper:
        continue
    try:
        window_image = image[y_lower:y_upper, x_left:x_right, :]
        window_image = np.transpose(window_image, (2, 0, 1))
    except:
        continue

    window_image = torch.from_numpy(window_image).float()
    window_image = F.interpolate(window_image.unsqueeze(0), size=tile, mode="bilinear", align_corners=False).squeeze(0)

    polygon_shifted = np.array([[y-y_lower, x-x_left] for y, x in region.coords])
    mask = polygon2mask(window_image.shape[1:], polygon_shifted)
    mask = torch.from_numpy(mask).unsqueeze(0).float()
    mask = mask[:, :tile, :tile]
    window_image_masked = window_image*mask
    
    transposed = window_image.numpy().transpose(1, 2, 0).astype(np.uint8)
    hed = rgb2hed(transposed)
    null = np.zeros_like(transposed[:, :, 0])
    hematoxylin_transposed = hed2rgb(np.stack((hed[:, :, 0], null, null), axis=-1))
    hematoxylin = torch.from_numpy((hematoxylin_transposed * 255).astype(np.uint8)).permute(2, 0, 1).float()
    hematoxylin_masked = hematoxylin * mask

    images[cell_id] = torch.cat([
        torch.cat([window_image, window_image_masked], dim=2),
        torch.cat([hematoxylin, hematoxylin_masked], dim=2)
    ], dim=1)
    centroids[cell_id] = region.centroid


# # SHEST

# In[12]:


raw_directory = '/data0/'
directory = '/data0/crp/'

args_common = dict(
    raw_directory = raw_directory,
    directory = directory,
    organ = "lung",
    batch_size = 1024,
    split = 0,
    epochs = 40,
    lr = 0.01,
    mode = "infer",
)


# In[13]:


data = Images(argparse.Namespace(**args_common), pixel_size, centroids, images)


# In[14]:


get_ipython().run_cell_magic('time', '', 'args_model = argparse.Namespace(\n    **args_common,\n    platform = "Xenium_Prime",\n    cell_type = "cell_type",\n    sources = [\n        "10X",\n        "SMC",\n        "SMC",\n    ],\n    samples = [\n        "Human_Lung_Cancer",\n        "03320",\n        "03331",\n    ],\n)\n\nconfig_model = Config(args_model)\n')


# In[15]:


modeling = Modeling(args_model, config_model, data)


# In[19]:


modeling.infer()


# In[20]:


adata_inferred = modeling.adata_inferred
adata_inferred


# In[21]:


colors_predicted = adata_inferred.obs['cell_type'].map(config_model.palette_type)
common_cells = set(colors_predicted.index) & set(images.keys())


# In[22]:


adata_inferred.obs['cell_type'].value_counts()


# In[23]:


stem = '.'.join(wsi.split('/')[-1].split('.')[:-1])
stem


# In[24]:


adata_inferred.write_h5ad(f"/data0/crp/results/cell_type_prediction/{stem}.h5ad")


# In[25]:


segments = []
colors = []
print("Selecting the cells from the prediction ...")
for region in tqdm(regions):
    cell_id = str(region.label)
    if cell_id in common_cells:
        color_predicted = colors_predicted.loc[cell_id]
        if not pd.isna(color_predicted):
            contours = find_contours(region.image, 0.5)
            for contour in contours:
                contour_shifted = contour + region.bbox[:2]
                segments.append(contour_shifted[:, ::-1])
                colors.append(color_predicted)


# In[26]:


get_ipython().run_cell_magic('time', '', '\nheight = image.shape[0]\nwidth = image.shape[1]\n\nif width >= height:\n    fig, ax = plt.subplots(figsize=(200*width//width, 200*height//width))\nelse:\n    fig, ax = plt.subplots(figsize=(200*width//height, 200*height//height))\n\nax.imshow(image)    \nax.add_collection(LineCollection(segments, colors=colors, linewidths=1))\nax.set_xlim(0, width)\nax.set_ylim(height, 0)\n\nplt.savefig(f"/data0/crp/results/cell_type_prediction/{stem}.png", bbox_inches=\'tight\')\nplt.close()\n')


# In[ ]:




