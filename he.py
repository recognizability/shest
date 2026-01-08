import os
import sys
import argparse
import json
import gc
from tqdm import tqdm
from glob import glob
import math
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd

import openslide
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from skimage.measure import regionprops, find_contours
from skimage.draw import polygon2mask
from skimage.color import rgb2hed, hed2rgb
from cellpose import models, io

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import n_cores, generator, tile, Config, lower_micrometer, upper_micrometer
from data import Images, quadruple_tile
from model import Modeling
from main import set_args

args, remaining = set_args()
args.batch_size = 1024
args.split = 0
args.mode = "infer"

#wsi = f"/data0/HE/LUAD/TCGA_LUAD/TCGA-55-A48X-01Z-00-DX1.A46C6373-8458-4D55-88C3-4C70A05F9F47.svs"
parser = argparse.ArgumentParser()
parser.add_argument("--wsi", type=str, help="Path of a whole slide image file")
args_additional = parser.parse_args(remaining)
args.wsi = args_additional.wsi
stem = '.'.join(args.wsi.split('/')[-1].split('.')[:-1])
slide = openslide.OpenSlide(args.wsi)
pixel_size = float(slide.properties['openslide.mpp-x'])

print("Loading raw image ...", end='')
image_raw = slide.read_region((0, 0), 0, slide.level_dimensions[0])
print("done.")
image = np.array(image_raw)[:, :, :3]

model = models.CellposeModel(gpu=True)
npy = args.directory + f"results/cell_type_prediction/{stem}.npy"
if not os.path.exists(npy):
    print("Segmenting the nuclei of cells ...", end='')
    masks, flows, styles = model.eval(image)
    np.save(npy, masks)
    print("done.")
else:
    print("Loading the segmentation masks ...", end='')
    masks = np.load(npy)
    print("done.")

regions = regionprops(masks)
print(len(regions), "cell masks are prepared.")

lower = int(math.ceil(lower_micrometer / pixel_size))
upper = int(math.ceil(upper_micrometer / pixel_size))
half = upper // 2

coords = region.coords.astype(np.int32)
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

    polygon_shifted = np.array([[
        int(round(tile*(y_coord - y_lower) / upper)), 
        int(round(tile*(x_coord - x_left) / upper))
    ] for y_coord, x_coord in coords])
    images[cell_id] = quadruple_tile(window_image, polygon_shifted)
    centroids[cell_id] = (y, x)

print(len(images), "cell images are prepared.")

config_model = Config(args)
data = Images(args, pixel_size, centroids, images)
modeling = Modeling(args, config_model, data)

adata_inferred = modeling.adata_inferred
colors_predicted = adata_inferred.obs['cell_type'].map(config_model.palette_type)
common_cells = set(colors_predicted.index) & set(images.keys())
adata_inferred.obs['cell_type'].value_counts()

adata_inferred.write_h5ad(args.directory + f"results/cell_type_prediction/{stem}.h5ad")

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

height = image.shape[0]
width = image.shape[1]
if width >= height:
    fig, ax = plt.subplots(figsize=(200*width//width, 200*height//width))
else:
    fig, ax = plt.subplots(figsize=(200*width//height, 200*height//height))

ax.imshow(image)    
ax.add_collection(LineCollection(segments, colors=colors, linewidths=1))
ax.set_xlim(0, width)
ax.set_ylim(height, 0)
plt.savefig(args.directory + f"/results/cell_type_prediction/{stem}.png", bbox_inches='tight')
plt.close()
