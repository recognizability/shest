import os
import numpy as np
import pandas as pd
from scipy import sparse

import spatialdata as sd
from spatialdata.transformations import get_transformation
import spatialdata_plot
from spatialdata_io import xenium

import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

n_cores = max(mp.cpu_count()-2, 1)
pixel_size = 0.2125  # micrometers per pixel
lower_bound = int(4 // pixel_size) #micrometer // pixel
upper_bound = int(15 // pixel_size) #micrometer // pixel
crop_size = upper_bound

def inverse_affine_transform(x_pixel, y_pixel):
    x_pixel = np.atleast_1d(x_pixel)
    y_pixel = np.atleast_1d(y_pixel)
    pixel_coords = np.stack([x_pixel, y_pixel], axis=1)
    ones = np.ones((pixel_coords.shape[0], 1))
    three_dimension = np.hstack([pixel_coords, ones]).T
    inverse_affine = np.linalg.inv(affine)
    transformed = inverse_affine @ three_dimension
    x_transformed, y_transformed = transformed[:2]
    if x_transformed.size == 1:
        return x_transformed[0], y_transformed[0]
    else:
        return x_transformed, y_transformed

def crop_he_image(cell_id):
    centroid = cell_boundaries.loc[cell_id, "geometry"].centroid
    xenium_x_um, xenium_y_um = centroid.x, centroid.y
    xenium_x_px, xenium_y_px = xenium_x_um / pixel_size, xenium_y_um / pixel_size
    he_x, he_y = inverse_affine_transform(xenium_x_px, xenium_y_px)
    he_x, he_y = round(he_x), round(he_y)
    half = crop_size // 2
    x_min = int(he_x - half)
    x_max = int(he_x + half)
    y_min = int(he_y - half)
    y_max = int(he_y + half)

    image_path = f'{processed_path}he_{crop_size}_micrometer/'
    os.makedirs(image_path, exist_ok=True)
    if 0 <= x_min and x_max < image_width and 0 <= y_min and y_max < image_height:
        cropped_image = he_image_array[:, y_min:y_max, x_min:x_max]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(cropped_image.transpose(1, 2, 0)) # (c, y, x) to (y, x, c) 
        ax.axis("off")   
        fig.savefig(f"{processed_path}he_{crop_size}_micrometer/{cell_id}.png", bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)
    
def cell_area_filter():
    bounds = sdata.shapes['cell_boundaries'].bounds
    len_raw = len(bounds)
    bounds = bounds / pixel_size
    bounds['width'] = bounds.apply(lambda row: row['maxx'] - row['minx'], axis=1)
    bounds['height'] = bounds.apply(lambda row: row['maxy'] - row['miny'], axis=1)

    fig, ax = plt.subplots(1,2)
    sns.violinplot(bounds, x='width', ax=ax[0])
    sns.violinplot(bounds, x='height', ax=ax[1])
    fig.tight_layout()
    fig.savefig(f"{processed_path}violinplot_width_height.png", bbox_inches="tight")
    plt.close(fig)
    
    filtered_cells = bounds[
        (lower_bound <= bounds['width']) & 
        (bounds['width'] < upper_bound) &
        (lower_bound <= bounds['height']) & 
        (bounds['height'] < upper_bound)
    ].index
    print(f"{len(filtered_cells)} cells are selected from {len_raw} cells.")
    
    return filtered_cells

def crop_cells(cell_ids):
    pool = Pool(n_cores)
    expression_dfs = pool.map(crop_he_image, tqdm(cell_ids))
    pool.close()
    pool.join()

def prepare_sdata(path):
    path_zarr = path + "data.zarr" 
    if not os.path.exists(path_zarr):
        sdata = xenium(
            path=path,
            n_jobs=n_cores,
        )
        print("Saving zarr ...")
        sdata.write(path_zarr) 
        print('Done.')
    else:
        print("Loading the zarr ...", end=' ')
        sdata = sd.SpatialData.read(path_zarr)
        print('done.')
    return sdata

platform = 'Xenium_Prime'
sample = 'Human_Lung_Cancer'
raw_path = f"/data0/{platform}/{sample}/"
sdata = prepare_sdata(raw_path)
affine = get_transformation(sdata.images['he_image']).to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
cell_boundaries = sdata.shapes["cell_boundaries"]

processed_path = f"/data0/crp/dataset/{platform}/{sample}/"
os.makedirs(processed_path, exist_ok=True)

'''
H&E annotation
'''
he_annotation = pd.read_csv(processed_path + f"annotation/merged_output.csv")
he_annotation.set_index("cell_id")[["group"]]
he_annotation.to_csv(processed_path + "annotation/he_annotation.csv")
he_annotation["cell_id"] = he_annotation["cell_id"].astype(str)

adata = sdata.tables["table"]
he_image = sdata.images["he_image"]["scale0"]
he_image_array = he_image["image"].values 
image_channels, image_height, image_width = he_image_array.shape  # (C, Y, X)

annotated_cell_ids = he_annotation["cell_id"].unique()
filtered_cell_ids = cell_area_filter()
cell_ids = set(annotated_cell_ids) & set(filtered_cell_ids)
crop_cells(cell_ids)
