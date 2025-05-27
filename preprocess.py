import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import torchvision.transforms as transforms

import spatialdata as sd
from spatialdata.transformations import get_transformation
import spatialdata_plot
from spatialdata_io import xenium

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import tacco as tc

from config import n_cores, seed, set_seed

sc.settings.n_jobs = n_cores

class Preprocessing():
    def __init__(self, args, config):
        self.raw_directory = args.raw_directory
        self.sample = args.sample
        self.stem_file = config.stem_file
        self.directory = args.directory
        self.sc_annotate = args.sc_annotate
        self.cell_types = config.cell_types
        self.cell_subtypes = config.cell_subtypes
        self.subtype_to_type = {subtype: category for category, subtypes in self.cell_types.items() for subtype in subtypes}

        self.pixel_size = json.load(open(self.raw_directory + config.stem_directory + "experiment.xenium"))['pixel_size'] # micrometers per pixel
        self.lower = 3 #micrometers
        self.upper = 18 #micrometers
        self.filter = args.filter

        self.sdata = self._prepare_sdata(self.raw_directory + config.stem_directory)
        self.affine = get_transformation(self.sdata.images['he_image']).to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
        self.nucleus_boundaries = self.sdata.shapes["nucleus_boundaries"]
        self.adata = self.sdata.tables['table']

        self.processing_directory = self.directory + 'dataset/' + config.stem_directory
        os.makedirs(self.processing_directory, exist_ok=True)

        self.he_image_array = self.sdata.images["he_image"]["scale0"]["image"].values #y, x, c
        self.image_channels, self.image_height, self.image_width = self.he_image_array.shape

        self.annotated_cell_ids = None
        self.image_ids = None
        self.images = None
        self.cell_ids = None
        
        self.sc_annotation()
        self.cell_area_filter()
        self.annotation()

    def _prepare_sdata(self, path):
        path_zarr = path + "data.zarr" 
        if not os.path.exists(path_zarr):
            sdata = xenium(path=path, n_jobs=n_cores)
            print("Saving zarr ...")
            sdata.write(path_zarr) 
            print('done.')
        else:
            print("Loading the zarr ...", end=' ')
            sdata = sd.SpatialData.read(path_zarr)
            print('done.')
        return sdata

    def _single_cell_reference(self):
        if 'lung' in self.sample.lower():
            print("Loading LuCA single cell reference ... ", end='')
            ref = sc.read_h5ad(self.raw_directory + 'cz_sc_reference/dd538ee7-f5e4-49e9-9f1e-2a1ea5246cf4.h5ad')
            ref.index = ref.var.feature_name
            ref.var.index = ref.var.feature_name
            ref = ref[ref.obs['platform']!='Smart-seq2'].copy()
            print(ref.shape)
        return ref

    def sc_annotation(self, cell_subtype='cell_type_tumor'):
        os.makedirs(self.processing_directory + f'annotation/', exist_ok=True)
        sc_annotation_csv = self.processing_directory + f'annotation/sc_annotation_{cell_subtype}.csv'
        sc_annotation_h5ad = self.processing_directory + f'annotation/sc_annotation_{cell_subtype}.h5ad'
        if not (os.path.exists(sc_annotation_csv) and os.path.exists(sc_annotation_h5ad)) or self.sc_annotate:
            ref = self._single_cell_reference()
            print('Annotating types of the cells ... ')
            self.adata.obs[cell_subtype] = tc.tl.annotate(
                self.adata,
                ref,
                annotation_key=cell_subtype,
                assume_filtered_counts=True,
                remove_constant_genes=False,
            ).T.idxmax()
            self.adata.obs[cell_subtype].to_csv(sc_annotation_csv)
            print(len(self.adata.obs[cell_subtype].unique()), 'subtypes are annotated.')

            print('Setting neighbors for each cell ...')
            sc.pp.neighbors(self.adata, random_state=seed)
            print('Making UMAPs for each cell ...')
            sc.tl.umap(self.adata, random_state=seed)

            self.adata.write_h5ad(sc_annotation_h5ad)
        else: 
            self.adata = sc.read_h5ad(sc_annotation_h5ad)

        self.adata.obs.index = self.adata.obs['cell_id']
        inclusion = self.adata.obs[cell_subtype].isin(self.cell_subtypes)
        self.adata.obs.loc[inclusion, 'cell_subtype_expression'] = self.adata.obs.loc[inclusion, cell_subtype]
        self.adata.obs['cell_subtype_expression'] = pd.Categorical(self.adata.obs['cell_subtype_expression'], categories=self.cell_subtypes)
        self.adata.obs['cell_type_expression'] = self.adata.obs[cell_subtype].map(self.subtype_to_type)

    def _inverse_affine_transform(self, x_pixel, y_pixel):
        x_pixel = np.atleast_1d(x_pixel)
        y_pixel = np.atleast_1d(y_pixel)
        pixel_coords = np.stack([x_pixel, y_pixel], axis=1)
        ones = np.ones((pixel_coords.shape[0], 1))
        three_dimension = np.hstack([pixel_coords, ones]).T
        inverse_affine = np.linalg.inv(self.affine)
        transformed = inverse_affine @ three_dimension
        x_transformed, y_transformed = transformed[:2]
        if x_transformed.size == 1:
            return x_transformed[0], y_transformed[0]
        else:
            return x_transformed, y_transformed

    def cell_area_filter(self):
        bounds = self.nucleus_boundaries.bounds
        bounds['width'] = bounds['maxx'] - bounds['minx']
        bounds['height'] = bounds['maxy'] - bounds['miny']
        bounds['cell_type_expression'] = self.adata.obs.loc[bounds.index, 'cell_type_expression']
        bounds_melted = bounds.melt(id_vars='cell_type_expression', value_vars=['width', 'height'], var_name='Side', value_name='Length_(μm)')
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.violinplot(data=bounds_melted, y='cell_type_expression', x='Length_(μm)', hue='Side', split=True, inner='quart', order=list(self.cell_types.keys()), ax=ax)
        ax.axvline(self.upper, ls='--', c='gray')
        ax.axvline(self.lower, ls='--', c='gray')
        fig.tight_layout()
        fig.savefig(f"{self.directory}results/violinplot_{self.stem_file}.png", bbox_inches="tight")
        plt.close(fig)

        centroids = self.nucleus_boundaries.centroid
        centroids_x, centroids_y = self._inverse_affine_transform(centroids.x / self.pixel_size, centroids.y / self.pixel_size)
        bounds['centroid_x'] = np.round(centroids_x).astype(int)
        bounds['centroid_y'] = np.round(centroids_y).astype(int)
        bounds = bounds.loc[
            (self.lower <= bounds['width']) & (bounds['width'] < self.upper) &
            (self.lower <= bounds['height']) & (bounds['height'] < self.upper)
        ]

        print("Filtering rectangular cell regions ... ", end='')
        self.window_images = {}
        self.nucleus_images = {}
        for bound in bounds.itertuples():
            cell_id = bound.Index
            window = int(np.ceil(self.upper / self.pixel_size))
            x_start = bound.centroid_x - window//2
            y_start = bound.centroid_y - window//2
            x_end = bound.centroid_x + window//2
            y_end = bound.centroid_y + window//2
            if x_start < 0 or y_start < 0 or self.image_width <= x_end + 1 or self.image_height <= y_end + 1:
                continue
            self.window_images[cell_id] = self.he_image_array[:, y_start : y_end + 1, x_start : x_end + 1]
            crop_size = int(np.ceil(max(bound.width, bound.height) / self.pixel_size))
            x_start = bound.centroid_x - crop_size//2
            y_start = bound.centroid_y - crop_size//2
            x_end = bound.centroid_x + crop_size//2
            y_end = bound.centroid_y + crop_size//2
            self.nucleus_images[cell_id] = self.he_image_array[:, y_start : y_end + 1, x_start : x_end + 1]

        self.image_ids = list(self.window_images.keys())
        print(len(self.image_ids))

        print("Save the images ...")
        images_directory = os.path.join(self.processing_directory, 'images/')
        os.makedirs(images_directory, exist_ok=True)
        for cell_id in tqdm(self.image_ids):
            np.save(os.path.join(images_directory, f"{cell_id}_window.npy"), self.window_images[cell_id])
            np.save(os.path.join(images_directory, f"{cell_id}_nucleus.npy"), self.nucleus_images[cell_id])

    def annotation(self):
        he_annotation = pd.read_csv(self.processing_directory + f"annotation/merged_output.csv", index_col='cell_id')
        he_annotation['cell_type_morphology'] = he_annotation['group'].astype(str)
        self.adata.obs = self.adata.obs.merge(he_annotation['cell_type_morphology'], how='left', left_index=True, right_index=True)
        print(f"{len(self.adata.obs['cell_type_morphology'].dropna())} cells are annotated by thier morphologies.")
        inclusion_morphology = self.adata.obs['cell_type_morphology'].notna()

        inclusion_annotation = self.adata.obs['cell_type_expression'].astype(str) == self.adata.obs['cell_type_morphology'].astype(str)
        self.adata.obs.loc[inclusion_annotation, 'cell_type_annotation'] = self.adata.obs.loc[inclusion_annotation, 'cell_type_expression']
        self.annotated_cell_ids = self.adata.obs[self.adata.obs['cell_type_annotation'].notna()].index
        print(f'Only the {len(self.annotated_cell_ids)} cells are annotated by the morphology and the single cell reference')
        self.adata.obs.loc[inclusion_annotation, 'cell_subtype_annotation'] = self.adata.obs.loc[inclusion_annotation, 'cell_subtype_expression']

        self.cell_ids = sorted(list(set(self.annotated_cell_ids) & set(self.image_ids)))
        print(f"Only {len(self.cell_ids)} cells common to both annotation and area filtering are prepared.")
        self.adata.obs['cell_type'] = self.adata.obs['cell_type_annotation'].where(self.adata.obs.index.isin(self.cell_ids), other=np.nan)
        self.adata.obs['cell_subtype'] = self.adata.obs['cell_subtype_annotation'].where(self.adata.obs.index.isin(self.cell_ids), other=np.nan)

        self.adata.obs = self.adata.obs[['cell_subtype_expression', 'cell_type_expression', 'cell_type_morphology', 'cell_subtype_annotation', 'cell_type_annotation', 'cell_subtype', 'cell_type']]
        self.adata.raw = self.adata
        self.adata.write(self.processing_directory + f'annotation/adata.h5ad')
