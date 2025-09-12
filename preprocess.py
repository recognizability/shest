import os
import gc
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
import torch

import spatialdata as sd
from spatialdata.transformations import get_transformation
import spatialdata_plot
from spatialdata_io import xenium

from skimage.measure import regionprops, find_contours

from tqdm import tqdm
import joblib
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import tacco as tc

from config import n_cores, seed, set_seed

sc.settings.n_jobs = n_cores

class Preprocessing():
    def __init__(self, args, config, source, sample):
        self.raw_directory = args.raw_directory
        self.source = source
        self.sample = sample
        self.stem_file = f"{args.platform}_{source}_{sample}"
        self.directory = args.directory
        self.sc_annotate = args.sc_annotate
        self.organ = config.organ
        self.cell_types = config.cell_types
        self.cell_subtype = config.cell_subtype
        self.cell_subtypes = config.cell_subtypes
        self.subtype_to_type = {subtype: category for category, subtypes in self.cell_types.items() for subtype in subtypes}

        self.stem_directory = f"{args.platform}/{source}/{sample}/"
        self.pixel_size = json.load(open(self.raw_directory + self.stem_directory + "experiment.xenium"))['pixel_size'] # micrometers per pixel
        self.lower = 3 #micrometers
        self.upper = 18 #micrometers

        self.sdata = self._prepare_sdata()
        self.affine = get_transformation(self.sdata.images['he_image']).to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
        self.nucleus_boundaries = self.sdata.shapes["nucleus_boundaries"]
        adata_raw = self.sdata.tables['table']
        self.adata = adata_raw[:, adata_raw.var_names.isin(config.gene_panel)].copy() 
        self.adata.obs.index = self.adata.obs['cell_id']

        self.processing_directory = self.directory + 'dataset/' + self.stem_directory
        os.makedirs(self.processing_directory, exist_ok=True)

        self.he_image_array = self.sdata.images["he_image"]["scale0"]["image"].values.astype(np.float16) #y, x, c
        self.image_channels, self.image_height, self.image_width = self.he_image_array.shape

        self.annotated_cell_ids = None
        self.image_ids = None
        self.images = None
        self.cell_ids = None
        
        self.sc_annotation()
        self.cell_filter()
        self.annotation()

    def _prepare_sdata(self):
        path_zarr = self.directory + "dataset/" + self.stem_directory + "data.zarr"
        if os.path.exists(path_zarr):
            print(f"Loading the {path_zarr} ...", end=' ')
            sdata = sd.SpatialData.read(path_zarr)
            print('done.')
        else:
            path = self.raw_directory + self.stem_directory
            sdata = xenium(path=path, n_jobs=n_cores)
            print(f"Saving {path_zarr} ...")
            sdata.write(path_zarr) 
            print('done.')
        return sdata

    def _single_cell_reference(self):
        if self.organ == 'lung':
            print("Loading LuCA single cell reference ... ", end='')
            ref = sc.read_h5ad(self.raw_directory + 'cz_sc_reference/dd538ee7-f5e4-49e9-9f1e-2a1ea5246cf4.h5ad')
            ref = ref[ref.obs['platform']!='Smart-seq2'].copy()
        if self.organ == 'breast':
            print("Loading breast cancer single cell reference ... ", end='')
            ref = sc.read_h5ad(self.raw_directory + 'cz_sc_reference/966b60ee-b416-44bd-981c-817bfc476646.h5ad')
        if self.organ == 'skin':
            print("Loading skin cancer single cell reference ... ", end='')
            ref = sc.read_h5ad(self.raw_directory + 'cz_sc_reference/f6e35982-3bef-47fe-b14a-60d2e8965f20.h5ad')
            ref = ref[ref.obs['disease'].str.contains('melanoma', case=False, na=False) & ref.obs['tissue'].str.contains('skin', case=False, na=False)].copy()
        ref.index = ref.var.feature_name
        ref.var.index = ref.var.feature_name
        print(ref.shape)
        return ref

    def sc_annotation(self):
        os.makedirs(self.processing_directory + f'annotation/', exist_ok=True)
        sc_annotation_csv = self.processing_directory + f'annotation/sc_annotation_{self.cell_subtype}.csv'
        sc_annotation_h5ad = self.processing_directory + f'annotation/sc_annotation_{self.cell_subtype}.h5ad'
        if not (os.path.exists(sc_annotation_csv) and os.path.exists(sc_annotation_h5ad)) or self.sc_annotate:
            ref = self._single_cell_reference()
            print('Annotating types of the cells ... ')
            self.adata.obs[self.cell_subtype] = tc.tl.annotate(
                self.adata,
                ref,
                annotation_key=self.cell_subtype,
                assume_valid_counts=True,
                remove_constant_genes=False,
            ).T.idxmax()
            self.adata.obs[self.cell_subtype].to_csv(sc_annotation_csv)
            print(len(self.adata.obs[self.cell_subtype].unique()), 'subtypes are annotated.')

            print('Setting neighbors for each cell ...')
            sc.pp.neighbors(self.adata, random_state=seed)
            print('Making UMAPs for each cell ...')
            sc.tl.umap(self.adata, random_state=seed)

            self.adata.write_h5ad(sc_annotation_h5ad)
        else: 
            self.adata = sc.read_h5ad(sc_annotation_h5ad)

        self.adata.obs.index = self.adata.obs['cell_id']
        self.adata.obs = self.adata.obs.drop(columns='cell_id')
        inclusion = self.adata.obs[self.cell_subtype].isin(self.cell_subtypes)
        self.adata.obs.loc[inclusion, 'cell_subtype_expression'] = self.adata.obs.loc[inclusion, self.cell_subtype]
        self.adata.obs['cell_subtype_expression'] = pd.Categorical(self.adata.obs['cell_subtype_expression'], categories=self.cell_subtypes)
        self.adata.obs['cell_type_expression'] = self.adata.obs[self.cell_subtype].map(self.subtype_to_type)

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

    def cell_filter(self):
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
        bounds['centroid_x'] = np.round(centroids_x).astype(np.int32)
        bounds['centroid_y'] = np.round(centroids_y).astype(np.int32)
        bounds = bounds.loc[
            (self.lower <= bounds['width']) & (bounds['width'] < self.upper) &
            (self.lower <= bounds['height']) & (bounds['height'] < self.upper)
        ]

        print("Filtering rectangular cell regions ... ", end='')
        self.images = {}
        window = int(np.ceil(self.upper / self.pixel_size))

        for bound in tqdm(bounds.itertuples(), total=len(bounds)):
            cell_id = bound.Index
            cx = bound.centroid_x
            cy = bound.centroid_y
            x_start = cx - window//2
            y_start = cy - window//2
            x_end = cx + window//2 + 1
            y_end = cy + window//2 + 1
            if x_start < 0 or y_start < 0 or self.he_image_array.shape[2] <= x_end or self.he_image_array.shape[1] <= y_end:
                continue
            window_image = self.he_image_array[:, y_start:y_end, x_start:x_end]
            self.images[cell_id] = {
                'window_image': window_image,
                'nucleus': int(np.ceil(max(bound.width, bound.height) / self.pixel_size)), #without inverse affine transform
            }

        self.image_ids = list(self.images.keys())
        print(len(self.image_ids))

        images_directory = os.path.join(self.processing_directory, 'images/')
        os.makedirs(images_directory, exist_ok=True)
        images_file = images_directory + f"images.pkl"
        if not os.path.exists(images_file):
            print("Save the images ...", end=' ')
            joblib.dump(self.images, os.path.join(images_file), compress=0)
            print("done.")

    def annotation(self):
        he_annotation_file = self.processing_directory + f"annotation/he_annotation.csv"
        if os.path.exists(he_annotation_file):
            he_annotation = pd.read_csv(he_annotation_file, index_col='cell_id')
            he_annotation['cell_type_morphology'] = he_annotation['group'].astype(str)
            self.adata.obs = self.adata.obs.merge(he_annotation['cell_type_morphology'], how='left', left_index=True, right_index=True)
            print(f"{len(self.adata.obs['cell_type_morphology'].dropna())} cells are annotated by thier morphologies.")

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
