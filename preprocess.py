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
import shapely
import torch.nn.functional as F
from skimage.draw import polygon2mask
from skimage.color import rgb2hed, hed2rgb

from tqdm import tqdm
import joblib

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
        self.lower = 2 #micrometers
        self.upper = 16 #micrometers

        self.sdata = self._prepare_sdata()
        self.affine = get_transformation(self.sdata.images['he_image']).to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
        self.nucleus_boundaries = self.sdata.shapes["nucleus_boundaries"]
        adata_raw = self.sdata.tables['table']
        self.adata = adata_raw[:, adata_raw.var_names.isin(config.gene_panel)].copy() 
        self.adata.obs.index = self.adata.obs['cell_id']

        self.processing_directory = self.directory + 'dataset/' + self.stem_directory
        os.makedirs(self.processing_directory, exist_ok=True)

        self.he_image = self.sdata.images["he_image"]["scale0"]["image"].values
        self.image_height, self.image_width = self.he_image.shape[1:]
        self.save_image = args.save_image

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
            print("Loading the sdata ...", end=' ')
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

            print("Saving the annotations ... ", end='')
            self.adata.write_h5ad(sc_annotation_h5ad)
            print("done.")
        else: 
            print("Loading the annotations ... ", end='')
            self.adata = sc.read_h5ad(sc_annotation_h5ad)
            print("done.")

        self.adata.obs.index = self.adata.obs['cell_id']
        self.adata.obs = self.adata.obs.drop(columns='cell_id')
        inclusion = self.adata.obs[self.cell_subtype].isin(self.cell_subtypes)
        self.adata.obs.loc[inclusion, 'cell_subtype_expression'] = self.adata.obs.loc[inclusion, self.cell_subtype]
        self.adata.obs['cell_subtype_expression'] = pd.Categorical(self.adata.obs['cell_subtype_expression'], categories=self.cell_subtypes, ordered=True)
        self.adata.obs['cell_type_expression'] = self.adata.obs[self.cell_subtype].map(self.subtype_to_type)
        self.adata.obs['cell_type_expression'] = pd.Categorical(self.adata.obs['cell_type_expression'], categories=self.cell_types.keys(), ordered=True)

    def _transform(self, polygon):
        x, y = polygon.exterior.xy
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x = x / self.pixel_size
        y = y / self.pixel_size

        coords = np.stack([x, y], axis=1)
        ones = np.ones((coords.shape[0], 1))
        three_dimension = np.hstack([coords, ones]).T
        inverse_affine = np.linalg.inv(self.affine)
        transformed = inverse_affine @ three_dimension
        x, y = transformed[:2]

        coords = np.stack([x, y], axis=1)
        return shapely.geometry.Polygon(coords).exterior

    def cell_filter(self):
        images_directory = os.path.join(self.processing_directory, 'images/')
        os.makedirs(images_directory, exist_ok=True)
        images_file = images_directory + f"images.pkl"
        if not os.path.exists(images_file) or self.save_image:
            side = 224
            tile_size = side // 2
            lower = int(round(self.lower/self.pixel_size))
            upper = int(round(self.upper/self.pixel_size))
            half = upper // 2
            exteriors = self.nucleus_boundaries.geometry.apply(lambda polygon: self._transform(polygon))
            self.images = {}
            print("Filtering and preparing cell images ...")
            for cell_id, polygon in tqdm(exteriors.items(), total=len(exteriors)):
                x, y = polygon.xy
                if np.min(x) < 0 or np.max(x) >= self.image_width or np.min(y) < 0 or np.max(y) >= self.image_height:
                    continue
                nucleus = max(np.max(y) - np.min(y), np.max(x) - np.min(x))
                nucleus = int(round(tile_size*nucleus/upper))
                if nucleus < lower or nucleus >= upper:
                    continue
                x_centroid = int(round(polygon.centroid.x))
                y_centroid = int(round(polygon.centroid.y))
                y_lower = y_centroid - half
                y_upper = y_centroid + half
                x_left = x_centroid - half
                x_right = x_centroid + half
                if y_lower < 0 or y_upper >= self.image_height or x_left < 0 or x_right >= self.image_width:
                    continue

                window_image = self.he_image[:, y_lower:y_upper, x_left:x_right].copy()
                window_image = torch.from_numpy(window_image).float()
                window_image = F.interpolate(window_image.unsqueeze(0), size=tile_size, mode="bilinear", align_corners=False).squeeze(0)

                polygon_shifted = [(
                    int(round(tile_size*(y - y_lower)/upper)),
                    int(round(tile_size*(x - x_left)/upper)),
                ) for x, y in polygon.coords]
                mask = polygon2mask(window_image.shape[1:], polygon_shifted)
                mask = torch.from_numpy(mask).unsqueeze(0).float()
                window_image_masked = window_image*mask

                center = window_image.shape[1] // 2
                nucleus_image = window_image[:, center-nucleus//2:center+nucleus//2+1, center-nucleus//2:center+nucleus//2+1]
                nucleus_image = F.interpolate(nucleus_image.unsqueeze(0), size=tile_size, mode="bilinear", align_corners=False).squeeze(0)
                nucleus_image_masked = window_image_masked[:, center-nucleus//2:center+nucleus//2+1, center-nucleus//2:center+nucleus//2+1]
                nucleus_image_masked = F.interpolate(nucleus_image_masked.unsqueeze(0), size=tile_size, mode="bilinear", align_corners=False).squeeze(0)

                self.images[cell_id] = torch.cat([
                    torch.cat([window_image, window_image_masked], dim=2),
                    torch.cat([nucleus_image, nucleus_image_masked], dim=2)
                ], dim=1)

            print("Saving the images ...", end=' ')
            joblib.dump(self.images, os.path.join(images_file), compress=0)
            print("done.")
        else:
            print("Loading the images ...", end=' ')
            with open(file=os.path.join(images_file), mode='rb') as f:
                self.images = joblib.load(f)
            print("done.")

        self.image_ids = list(self.images.keys())
        print(len(self.image_ids), "images are prepared.")

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
