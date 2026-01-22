import os
import gc
import sys
import argparse
import joblib
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

import openslide
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from skimage.measure import regionprops, find_contours
from skimage.draw import polygon2mask
from cellpose import models, io, utils
from shapely.geometry import Polygon, mapping
import cv2
import json

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
parser.add_argument("--save_colored_image", action="store_true", help="Save a colored image file")
args_additional = parser.parse_args(remaining)
args.wsi = args_additional.wsi
args.save_colored_image = args_additional.save_colored_image
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
adata_file = path_stem.with_name(path_stem.name + ".h5ad")
if not os.path.exists(adata_file):
    data = Images(args, pixel_size, centroids, images)
    modeling = Modeling(args, config_model, data)
    adata_inferred = modeling.adata_inferred
    print("Saving the expressions and the cell types ...", end=' ')
    adata_inferred.write_h5ad(adata_file)
    print("done.")
else:
    print("Loading the adata file ...", end=' ')
    adata_inferred = sc.read_h5ad(adata_file)
    print("done.")

print(adata_inferred)
print(adata_inferred.obs['cell_type'].value_counts())
colors_predicted = adata_inferred.obs['cell_type'].map(config_model.palette_type)
common_cells = set(colors_predicted.index) & set(images.keys())

print("Selecting the cells from the prediction ...", end=' ')
segments = []
colors = []
polygons = []
cell_types = []
for region in tqdm(regions):
    cell_id = str(region.label)
    if not cell_id in common_cells:
        continue
    color = colors_predicted.loc[cell_id]
    if pd.isna(color):
        continue
    mask = (region.image.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    colors.append(color)
    contour = max(contours, key=len)
    contour = contour.reshape(-1, 2)
    segment = contour + np.array([region.bbox[1], region.bbox[0]])
    if len(segment) < 4:
        continue
    segments.append(segment)
    polygon = Polygon(segment)
    polygons.append(polygon)
    cell_type = adata_inferred.obs.loc[cell_id, 'cell_type']
    cell_types.append(cell_type)
print("done.")
print(len(segments), "segments are prepared.")

print("Serializing geojson ...", end=' ')
features = []
for i, polygon in tqdm(enumerate(polygons), total=len(polygons)):
    if not polygon.is_valid or polygon.is_empty:
        continue
    rgb = [int(colors[i][c:c+2], 16) for c in (1, 3, 5)]
    features.append({
        "type": "Feature",
        "geometry": mapping(polygon),
        "properties": {
            "cell_type": cell_types[i],
            "color": rgb,
        },
    })
print(len(features), "features are prepared.")
geojson = {
    "type": "FeatureCollection",
    "features": features
}
print("Saving a geojson file ...", end=' ')
with open(path_stem.with_name(path_stem.name + ".geojson"), "w") as f:
    json.dump(geojson, f)
print("done.")

if args.save_colored_image:
    height = image.shape[0]
    width = image.shape[1]
    scale = 100
    if width >= height:
        fig, ax = plt.subplots(figsize=(scale*width//width, scale*height//width))
    else:
        fig, ax = plt.subplots(figsize=(scale*width//height, scale*height//height))
    ax.imshow(image)    
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    print("Saving an image colored ...", end=' ')
    plt.savefig(path_stem.with_name(path_stem.name + ".png"), bbox_inches="tight")
    plt.close()
    print("done.")
