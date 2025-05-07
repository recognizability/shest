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

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import tacco as tc

from config import n_cores, seed, set_seed

sc.settings.n_jobs = n_cores

class Preprocessing():
    def __init__(self, args, config):
        self.raw_directory = args.raw_directory
        self.stem_file = config.stem_file
        self.directory = args.directory
        self.force_annotate = args.force_annotate
        self.force_categorize = args.force_categorize
        self.cell_type = args.cell_type
        self.cell_types = config.cell_types

        self.pixel_size = json.load(open(self.raw_directory + config.stem_directory + "experiment.xenium"))['pixel_size'] # micrometers per pixel
        self.lower_bound = int(4 // self.pixel_size) #micrometer/(micrometer/pixel)
        self.upper_bound = int(18 // self.pixel_size) #micrometer/(micrometer/pixel)
        self.crop_size = self.upper_bound

        self.sdata = self._prepare_sdata(self.raw_directory + config.stem_directory)
        self.affine = get_transformation(self.sdata.images['he_image']).to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
        self.cell_boundaries = self.sdata.shapes["cell_boundaries"]
        self.adata = self.sdata.tables['table']

        self.processing_directory = self.directory + 'dataset/' + config.stem_directory
        os.makedirs(self.processing_directory, exist_ok=True)

        self.he_image_array = self.sdata.images["he_image"]["scale0"]["image"].values 
        self.image_channels, self.image_height, self.image_width = self.he_image_array.shape #(c, y, x)

        self.image_directory = None
        self.annotated_cell_ids = None
        self.filtered_cell_ids = None
        self.cell_ids = None

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
        if self.organ == 'Lung':
            print("Loading LuCA single cell reference ... ", end='')
            ref = sc.read_h5ad(self.raw_directory + 'cz_sc_reference/dd538ee7-f5e4-49e9-9f1e-2a1ea5246cf4.h5ad')
            ref.index = ref.var.feature_name
            ref.var.index = ref.var.feature_name
            ref = ref[ref.obs['platform']!='Smart-seq2'].copy()
            print(ref.shape)
        return ref

    def annotation(self, cell_subtype='cell_type_tumor'):
        he_annotation = pd.read_csv(self.processing_directory + f"annotation/merged_output.csv")
        he_annotation = he_annotation.set_index("cell_id")[["group"]]
        he_annotation.to_csv(self.processing_directory + "annotation/he_annotation.csv")
        he_annotation['Cell_type_HE'] = he_annotation['group'].astype(str)
        print(f"{he_annotation.shape[0]} cells are annotated by thier morphologies.")
        
        sc_annotation_csv = self.processing_directory + f'annotation/sc_annotation_{cell_subtype}.csv'
        sc_annotation_h5ad = self.processing_directory + f'annotation/sc_annotation_{cell_subtype}.h5ad'
        if not (os.path.exists(sc_annotation_csv) and os.path.exists(sc_annotation_h5ad)) or self.force_annotate:
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

            adata_file = self.processing_directory + f'annotation/adata_he{self.crop_size}.h5ad'
            if not os.path.exists(adata_file) or self.force_categorize:
                self.adata.obs.index = self.adata.obs['cell_id']
                self.adata.obs['Cell_subtype_ST'] = self.adata.obs[cell_subtype]
                self.adata.obs['Cell_type_ST'] = self.adata.obs[cell_subtype].map({cell: group for group, subtypes in self.cell_types.items() for cell in subtypes})#.fillna('Other')
                self.adata.obs = self.adata.obs.merge(he_annotation['Cell_type_HE'], how='left', left_index=True, right_index=True)
                self.adata.obs['Cell_type_HE'] = self.adata.obs['Cell_type_HE']#.fillna('Other')
                self.adata.obs['Cell_type'] = self.adata.obs.loc[self.adata.obs['Cell_type_ST'].astype(str) == self.adata.obs['Cell_type_HE'].astype(str), 'Cell_type_HE']
                self.adata.obs = self.adata.obs[['Cell_subtype_ST', 'Cell_type_ST', 'Cell_type_HE', 'Cell_type']]
                self.adata.raw = self.adata

                self.adata.write(adata_file)
            else:
                self.adata = sc.read_h5ad(adata_file)

        self.annotated_cell_ids = self.adata.obs[self.adata.obs[self.cell_type].notna()].index
        print(f'{len(self.annotated_cell_ids)} cells are used to make their H&E images')

    def cell_area_filter(self):
        bounds = self.cell_boundaries.bounds
        len_raw = len(bounds)
        bounds = bounds / self.pixel_size #in pixel
        bounds['width'] = bounds.apply(lambda row: row['maxx'] - row['minx'], axis=1)
        bounds['height'] = bounds.apply(lambda row: row['maxy'] - row['miny'], axis=1)
        bounds = bounds.merge(self.adata.obs['Cell_type'], how='left', left_index=True, right_index=True)
        bounds_melted = bounds.melt(id_vars='Cell_type', value_vars=['width', 'height'], var_name='Length', value_name='Length_(μm)')
        bounds_melted["Length_(μm)"] *= self.pixel_size #in micrometer


        fig = plt.figure(figsize=(4, 4))
        sns.violinplot(bounds_melted, x='Length_(μm)', y='Cell_type', hue='Length', split=True, inner='quart')
        plt.axvline(x=self.upper_bound*self.pixel_size, linestyle='--', color='gray') #in micrometer
        plt.axvline(x=self.lower_bound*self.pixel_size, linestyle='--', color='gray') #in micrometer
        fig.tight_layout()
        fig.savefig(self.directory + f"results/violinplot_{self.stem_file}.png", bbox_inches="tight")
        plt.close(fig)
        
        self.filtered_cell_ids = bounds[
            (self.lower_bound <= bounds['width']) & 
            (bounds['width'] < self.upper_bound) &
            (self.lower_bound <= bounds['height']) & 
            (bounds['height'] < self.upper_bound)
        ].index
        print(f"{len(self.filtered_cell_ids)} cells are selected from {len_raw} cells.")

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

    def _crop_he_image(self, cell_id):
        centroid = self.cell_boundaries.loc[cell_id, "geometry"].centroid
        xenium_x_um, xenium_y_um = centroid.x, centroid.y
        xenium_x_px, xenium_y_px = xenium_x_um / self.pixel_size, xenium_y_um / self.pixel_size
        he_x, he_y = self._inverse_affine_transform(xenium_x_px, xenium_y_px)
        he_x, he_y = round(he_x), round(he_y)
        half = self.crop_size // 2
        x_min = int(he_x - half)
        x_max = int(he_x + half)
        y_min = int(he_y - half)
        y_max = int(he_y + half)

        if 0 <= x_min and x_max < self.image_width and 0 <= y_min and y_max < self.image_height:
            cropped_image = self.he_image_array[:, y_min:y_max, x_min:x_max]
            cropped_image = cropped_image.transpose(1, 2, 0) # (c, y, x) to (y, x, c)
            cropped_image = Image.fromarray(cropped_image)
            cropped_image.save(self.image_directory + f"{cell_id}.png")
        
    def crop_the_common_cells(self):
        self.image_directory = self.processing_directory + f'he{self.crop_size}/{self.cell_type}/'
        os.makedirs(self.image_directory, exist_ok=True)
        self.cell_ids = set(self.annotated_cell_ids) & set(self.filtered_cell_ids)
        print(f'{len(self.cell_ids)} cells are been preparing in {self.image_directory} directory for modeling')
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            list(tqdm(executor.map(self._crop_he_image, self.cell_ids, chunksize=len(self.cell_ids)//n_cores), total=len(self.cell_ids)))
