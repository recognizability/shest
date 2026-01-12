import os
import gc
import sys
import argparse
import joblib
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
from pathlib import Path

import openslide
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from skimage.measure import regionprops, find_contours
from skimage.draw import polygon2mask
from skimage.color import rgb2hed, hed2rgb
from cellpose import models, io

from config import n_cores, generator, tile, Config, lower_micrometer, upper_micrometer
from data import Images, quadruple_tile
from model import Modeling
from main import set_args

args, remaining = set_args()
args.batch_size = 1024
args.split = 0
args.mode = "infer"

parser = argparse.ArgumentParser()
parser.add_argument("--wsi", type=str, help="Path of a whole slide image file")
args_additional = parser.parse_args(remaining)
args.wsi = args_additional.wsi
stem = '.'.join(args.wsi.split('/')[-1].split('.')[:-1])
path_stem = Path(args.directory) / "he" / stem
path_stem.parent.mkdir(parents=True, exist_ok=True)

slide = openslide.OpenSlide(args.wsi)
pixel_size = float(slide.properties['openslide.mpp-x'])
lower = int(math.ceil(lower_micrometer / pixel_size))
upper = int(math.ceil(upper_micrometer / pixel_size))
half = upper // 2

print("Loading raw image ...", end=' ')
image_raw = slide.read_region((0, 0), 0, slide.level_dimensions[0])
print("done.")
print("Making it to an image array ...", end=' ')
image = np.array(image_raw)[:, :, :3]
print("done.")

masks_file = path_stem.with_name(path_stem.name + "_masks.npy")
if not os.path.exists(masks_file):
    model = models.CellposeModel(gpu=True)
    print("Segmenting the nuclei of cells ...", end=' ')
    masks, flows, styles = model.eval(image)
    np.save(masks_file, masks)
    print("done.")
else:
    print("Loading the segmentation masks ...", end=' ')
    masks = np.load(masks_file)
    print("done.")

regions = regionprops(masks)
print(len(regions), "cell masks are prepared.")

images_file = path_stem.with_name(path_stem.name + "_images.pkl")
centroids_file = path_stem.with_name(path_stem.name + "_centroids.pkl")
if (not os.path.exists(images_file)) or (not os.path.exists(centroids_file)):
    images = {}
    centroids = {}
    for region in tqdm(regions):
        cell_id = str(region.label)
        y_centroid, x_centroid = map(int, region.centroid)
        y_min, x_min, y_max, x_max = region.bbox
        nucleus = max(y_max - y_min, x_max - x_min)
        y_lower = y_centroid - half
        y_upper = y_centroid + half
        x_left = x_centroid - half
        x_right = x_centroid + half
        if y_lower < 0 or y_upper > image.shape[0] or x_left < 0 or x_right > image.shape[1] or nucleus < lower or nucleus > upper:
            continue
        try:
            window_image = image[y_lower:y_upper, x_left:x_right, :]
            window_image = np.transpose(window_image, (2, 0, 1))
        except:
            continue

        polygon_shifted = np.array([[y_coord - y_lower, x_coord - x_left] for y_coord, x_coord in region.coords])
        images[cell_id] = quadruple_tile(window_image, polygon_shifted, upper)
        centroids[cell_id] = (y_centroid, x_centroid)

    print("Saving the images and centroids ...", end=' ')
    joblib.dump(images, os.path.join(images_file), compress=0)
    joblib.dump(centroids, os.path.join(centroids_file), compress=0)
    print("done.")

else:
    print("Loading the images and centroids...", end=' ')
    with open(file=os.path.join(images_file), mode='rb') as f:
        images = joblib.load(f)
    with open(file=os.path.join(centroids_file), mode='rb') as f:
        centroids = joblib.load(f)
    print("done.")

print(len(images), "cell images are prepared.")

config_model = Config(args)
data = Images(args, pixel_size, centroids, images)
modeling = Modeling(args, config_model, data)

adata_inferred = modeling.adata_inferred
colors_predicted = adata_inferred.obs['cell_type'].map(config_model.palette_type)
common_cells = set(colors_predicted.index) & set(images.keys())
print(adata_inferred.obs['cell_type'].value_counts())
print("Saving the expressions and the cell types ...", end=' ')
adata_inferred.write_h5ad(path_stem.with_name(path_stem.name + ".h5ad"))
print("done.")

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
print("Saving the image colored ...", end=' ')
plt.savefig(path_stem.with_name(path_stem.name + ".png"), bbox_inches="tight")
plt.close()
print("done.")
