import os
import argparse
import json
import numpy as np
import pandas as pd
from scipy import sparse

import spatialdata as sd
from spatialdata.transformations import get_transformation
import spatialdata_plot
from spatialdata_io import xenium

from multiprocessing import Pool
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import tacco as tc

from config import n_cores, seed, set_seed, cell_types_lung

sc.settings.n_jobs = n_cores

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

    image_path = processed_path + f'he{crop_size}/'
    os.makedirs(image_path, exist_ok=True)
    if 0 <= x_min and x_max < image_width and 0 <= y_min and y_max < image_height:
        cropped_image = he_image_array[:, y_min:y_max, x_min:x_max]
        cropped_image = cropped_image.transpose(1, 2, 0) # (c, y, x) to (y, x, c)
        cropped_image = Image.fromarray(cropped_image)
        cropped_image.save(image_path + f"{cell_id}.png")
    
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
    fig.savefig(f"/data0/crp/results/violinplot_{args.platform}_{args.sample}.png", bbox_inches="tight")
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

def single_cell_reference():
    if organ == 'Lung':
        print("Loading LuCA single cell reference ... ", end='')
        ref = sc.read_h5ad('/data0/cz_sc_reference/dd538ee7-f5e4-49e9-9f1e-2a1ea5246cf4.h5ad')
        ref.index = ref.var.feature_name
        ref.var.index = ref.var.feature_name
        ref = ref[ref.obs['platform']!='Smart-seq2'].copy()
        print(ref.shape)
    return ref

def annotation(cell_subtype):
    he_annotation = pd.read_csv(processed_path + f"annotation/merged_output.csv")
    he_annotation = he_annotation.set_index("cell_id")[["group"]]
    he_annotation.to_csv(processed_path + "annotation/he_annotation.csv")
    he_annotation['Cell_type_HE'] = he_annotation['group'].astype(str)
    annotated_cell_ids = he_annotation.index
    
    adata = sdata.tables['table']
    sc_annotation_csv = processed_path + f'annotation/sc_annotation_{cell_subtype}.csv'
    sc_annotation_h5ad = processed_path + f'annotation/sc_annotation_{cell_subtype}.h5ad'
    if not (os.path.exists(sc_annotation_csv) and os.path.exists(sc_annotation_h5ad)) or args.force_annotate:
        ref = single_cell_reference()
        print('Annotating types of the cells ... ')
        adata.obs[cell_subtype] = tc.tl.annotate(
            adata,
            ref,
            annotation_key=cell_subtype,
            assume_valid_counts=True,
            remove_constant_genes=False,
        ).T.idxmax()
        adata.obs[cell_subtype].to_csv(sc_annotation_csv)
        print(len(adata.obs[cell_subtype].unique()), 'subtypes are annotated.')

        print('Setting neighbors for each cell ...')
        sc.pp.neighbors(adata, random_state=seed)
        print('Making UMAPs for each cell ...')
        sc.tl.umap(adata, random_state=seed)

        adata.write_h5ad(sc_annotation_h5ad)
    else: 
        adata = sc.read_h5ad(sc_annotation_h5ad)

        adata_file = processed_path + f'annotation/adata_he{crop_size}.h5ad'
        if not os.path.exists(adata_file) or args.force_categorize:
            adata.obs.index = adata.obs['cell_id']
            adata.obs['Cell_subtype_ST'] = adata.obs[cell_subtype]
            adata.obs['Cell_type_ST'] = adata.obs[cell_subtype].map({cell: group for group, subtypes in cell_types.items() for cell in subtypes})#.fillna('Other')
            adata.obs = adata.obs.merge(he_annotation['Cell_type_HE'], how='left', left_index=True, right_index=True)
            adata.obs['Cell_type_HE'] = adata.obs['Cell_type_HE']#.fillna('Other')
            adata.obs['Cell_type'] = adata.obs.loc[adata.obs['Cell_type_ST'].astype(str) == adata.obs['Cell_type_HE'].astype(str), 'Cell_type_HE']
            adata.obs = adata.obs[['Cell_subtype_ST', 'Cell_type_ST', 'Cell_type_HE', 'Cell_type']]

            adata.write(adata_file)

    return annotated_cell_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample information", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--directory", type=str, default="/data0/crp/dataset/", help="Directory of dataset")
    parser.add_argument("--platform", type=str, default="Xenium_Prime", help="Platform of spatial transcriptomics")
    parser.add_argument("--sample", type=str, default="Human_Lung_Cancer", help="Sample name")
    parser.add_argument("--force_annotate", action="store_true", help="If set, annotate again")
    parser.add_argument("--force_categorize", action="store_true", help="If set, categorize again")
    args = parser.parse_args()

    pixel_size = json.load(open(f"/data0/{args.platform}/{args.sample}/experiment.xenium"))['pixel_size'] # micrometers per pixel
    lower_bound = int(4 // pixel_size) #micrometer/(micrometer/pixel)
    upper_bound = int(15 // pixel_size) #micrometer/(micrometer/pixel)
    crop_size = upper_bound

    if 'lung' in args.sample or 'Lung' in args.sample:
        organ = 'Lung'
        cell_types = cell_types_lung

    raw_path = f"/data0/{args.platform}/{args.sample}/"
    sdata = prepare_sdata(raw_path)
    affine = get_transformation(sdata.images['he_image']).to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
    cell_boundaries = sdata.shapes["cell_boundaries"]

    processed_path = args.directory + f"{args.platform}/{args.sample}/"
    os.makedirs(processed_path, exist_ok=True)

    he_image = sdata.images["he_image"]["scale0"]
    he_image_array = he_image["image"].values 
    image_channels, image_height, image_width = he_image_array.shape #(c, y, x)

    annotated_cell_ids = annotation('cell_type_tumor')
    filtered_cell_ids = cell_area_filter()
    cell_ids = set(annotated_cell_ids) & set(filtered_cell_ids)
    crop_cells(cell_ids)
