import os
import argparse
import json
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
        self.lower = 4 #micrometers
        self.upper = args.upper #micrometers
        self.upper_string = f"upper{args.upper}"
        self.crop_size = int(self.upper // self.pixel_size) #pixels
        self.filter = args.filter

        self.sdata = self._prepare_sdata(self.raw_directory + config.stem_directory)
        self.affine = get_transformation(self.sdata.images['he_image']).to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
        self.cell_boundaries = self.sdata.shapes["cell_boundaries"]
        self.adata = self.sdata.tables['table']

        self.processing_directory = self.directory + 'dataset/' + config.stem_directory
        os.makedirs(self.processing_directory, exist_ok=True)

        self.he_image_array = self.sdata.images["he_image"]["scale0"]["image"].values #y, x, c
        self.image_channels, self.image_height, self.image_width = self.he_image_array.shape

        self.annotated_cell_ids = None
        self.image_ids = None
        self.cell_ids = None
        
        self.sc_annotation()
        self.cell_area_filter()
        self.annotation()

    def _prepare_sdata(self, path):
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
                assume_valid_counts=True,
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
        bounds = self.cell_boundaries.bounds
        bounds['width'] = bounds.apply(lambda row: row['maxx'] - row['minx'], axis=1)
        bounds['height'] = bounds.apply(lambda row: row['maxy'] - row['miny'], axis=1)
        bounds = bounds.merge(self.adata.obs['cell_type_expression'], how='left', left_index=True, right_index=True)
        bounds_melted = bounds.melt(id_vars='cell_type_expression', value_vars=['width', 'height'], var_name='Length', value_name='Length_(μm)')

        fig = plt.figure(figsize=(4, 4))
        sns.violinplot(bounds_melted, x='Length_(μm)', y='cell_type_expression', hue='Length', split=True, inner='quart', order=list(self.cell_types.keys()))
        plt.axvline(x=self.upper, linestyle='--', color='gray')
        plt.axvline(x=self.lower, linestyle='--', color='gray')
        fig.tight_layout()
        fig.savefig(self.directory + f"results/violinplot_{self.stem_file}_{self.upper_string}.png", bbox_inches="tight")
        plt.close(fig)
        
        bounds_ids = bounds[
            (self.lower <= bounds['width']) & 
            (bounds['width'] < self.upper) &
            (self.lower <= bounds['height']) & 
            (bounds['height'] < self.upper)
        ].index
        print(f"{len(bounds_ids)} cells are filtered by their area.")

        centroids = self.cell_boundaries.loc[bounds_ids, "geometry"].centroid
        centroids_x, centroids_y = self._inverse_affine_transform(centroids.x/self.pixel_size, centroids.y/self.pixel_size)
        centroids_x = centroids_x.round().astype(int)
        centroids_y = centroids_y.round().astype(int)
        half = int(self.crop_size // 2)
        coords = np.stack([centroids_x-half, centroids_x+half, centroids_y-half, centroids_y+half], axis=1)
        print("Making images to torch tensors ... ", end='')
        ids_images = {
            cell_id: torch.from_numpy(self.he_image_array[:, y_min:y_max, x_min:x_max])
            for cell_id, (x_min, x_max, y_min, y_max) in zip(bounds_ids, coords)
            if 0 <= x_min < x_max < self.image_width and 0 <= y_min < y_max < self.image_height
        }
        self.image_ids = ids_images.keys()
        print(len(self.image_ids))
        os.makedirs(self.processing_directory + f'images/', exist_ok=True)
        images_file = self.processing_directory + f"images/images_{self.upper_string}.pt"
        ids_file = self.processing_directory + f"images/image_ids_{self.upper_string}.json"
        print("Stacking the torch tensors ... ")
        images = torch.stack(list(ids_images.values()))
        print(images.shape)
        print("Save the tensor ...")
        torch.save(images, images_file)
        json.dump({cell_id: i for i, cell_id in enumerate(self.image_ids)}, open(ids_file, "w"))

    def annotation(self):
        he_annotation = pd.read_csv(self.processing_directory + f"annotation/merged_output.csv", index_col='cell_id')
        he_annotation['cell_type_morphology'] = he_annotation['group'].astype(str)
        self.adata.obs = self.adata.obs.merge(he_annotation['cell_type_morphology'], how='left', left_index=True, right_index=True)
        print(f"{len(self.adata.obs['cell_type_morphology'].dropna())} cells are annotated by thier morphologies.")
        inclusion_morphology = self.adata.obs['cell_type_morphology'].notna()
        self.adata.obs.loc[inclusion_morphology, 'cell_subtype_morphology'] = self.adata.obs.loc[inclusion_morphology, 'cell_subtype_expression']

        inclusion_annotation = self.adata.obs['cell_type_expression'].astype(str) == self.adata.obs['cell_type_morphology'].astype(str)
        self.adata.obs.loc[inclusion_annotation, 'cell_type_annotation'] = self.adata.obs.loc[inclusion_annotation, 'cell_type_morphology']
        self.annotated_cell_ids = self.adata.obs[self.adata.obs['cell_type_annotation'].notna()].index
        print(f'Only the {len(self.annotated_cell_ids)} cells are annotated by the morphology and the single cell reference')
        self.adata.obs.loc[inclusion_annotation, 'cell_subtype_annotation'] = self.adata.obs.loc[inclusion_annotation, 'cell_subtype_morphology']

        self.cell_ids = sorted(list(set(self.annotated_cell_ids) & set(self.image_ids)))
        print(f"Only {len(self.cell_ids)} cells common to both annotation and area filtering are prepared.")
        self.adata.obs['cell_type'] = self.adata.obs['cell_type_annotation'].where(self.adata.obs.index.isin(self.cell_ids), other=np.nan)
        self.adata.obs['cell_subtype'] = self.adata.obs['cell_subtype_annotation'].where(self.adata.obs.index.isin(self.cell_ids), other=np.nan)

        self.adata.obs = self.adata.obs[['cell_subtype_expression', 'cell_type_expression', 'cell_subtype_morphology', 'cell_type_morphology', 'cell_subtype_annotation', 'cell_type_annotation', 'cell_subtype', 'cell_type']]
        self.adata.raw = self.adata
        self.adata.write(self.processing_directory + f'annotation/adata.h5ad')
