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

class Preprocess():
    def __init__(self, platform, sample):
        self.platform = platform
        self.sample = sample
        self.pixel_size = json.load(open(f"/data0/{self.platform}/{self.sample}/experiment.xenium"))['pixel_size'] # micrometers per pixel
        self.lower_bound = int(4 // self.pixel_size) #micrometer/(micrometer/pixel)
        self.upper_bound = int(15 // self.pixel_size) #micrometer/(micrometer/pixel)
        self.crop_size = self.upper_bound

        if 'lung' in self.sample or 'Lung' in self.sample:
            self.organ = 'Lung'
            self.cell_types = cell_types_lung

        raw_path = f"/data0/{self.platform}/{self.sample}/"
        self.sdata = self._prepare_sdata(raw_path)
        self.affine = get_transformation(self.sdata.images['he_image']).to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
        self.cell_boundaries = self.sdata.shapes["cell_boundaries"]
        self.adata = self.sdata.tables['table']

        self.processed_path = args.directory + f"{self.platform}/{self.sample}/"
        os.makedirs(self.processed_path, exist_ok=True)

        self.he_image_array = self.sdata.images["he_image"]["scale0"]["image"].values 
        self.image_channels, self.image_height, self.image_width = self.he_image_array.shape #(c, y, x)

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

    def single_cell_reference():
        if self.organ == 'Lung':
            print("Loading LuCA single cell reference ... ", end='')
            ref = sc.read_h5ad('/data0/cz_sc_reference/dd538ee7-f5e4-49e9-9f1e-2a1ea5246cf4.h5ad')
            ref.index = ref.var.feature_name
            ref.var.index = ref.var.feature_name
            ref = ref[ref.obs['platform']!='Smart-seq2'].copy()
            print(ref.shape)
        return ref

    def annotation(self, cell_subtype='cell_type_tumor'):
        he_annotation = pd.read_csv(self.processed_path + f"annotation/merged_output.csv")
        he_annotation = he_annotation.set_index("cell_id")[["group"]]
        he_annotation.to_csv(self.processed_path + "annotation/he_annotation.csv")
        he_annotation['Cell_type_HE'] = he_annotation['group'].astype(str)
        self.annotated_cell_ids = he_annotation.index
        print(f"{len(self.annotated_cell_ids)} cells are annotated.")
        
        sc_annotation_csv = self.processed_path + f'annotation/sc_annotation_{cell_subtype}.csv'
        sc_annotation_h5ad = self.processed_path + f'annotation/sc_annotation_{cell_subtype}.h5ad'
        if not (os.path.exists(sc_annotation_csv) and os.path.exists(sc_annotation_h5ad)) or args.force_annotate:
            ref = single_cell_reference()
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

            adata_file = self.processed_path + f'annotation/adata_he{self.crop_size}.h5ad'
            if not os.path.exists(adata_file) or args.force_categorize:
                self.adata.obs.index = self.adata.obs['cell_id']
                self.adata.obs['Cell_subtype_ST'] = self.adata.obs[cell_subtype]
                self.adata.obs['Cell_type_ST'] = self.adata.obs[cell_subtype].map({cell: group for group, subtypes in self.cell_types.items() for cell in subtypes})#.fillna('Other')
                self.adata.obs = self.adata.obs.merge(he_annotation['Cell_type_HE'], how='left', left_index=True, right_index=True)
                self.adata.obs['Cell_type_HE'] = self.adata.obs['Cell_type_HE']#.fillna('Other')
                self.adata.obs['Cell_type'] = self.adata.obs.loc[self.adata.obs['Cell_type_ST'].astype(str) == self.adata.obs['Cell_type_HE'].astype(str), 'Cell_type_HE']
                self.adata.obs = self.adata.obs[['Cell_subtype_ST', 'Cell_type_ST', 'Cell_type_HE', 'Cell_type']]

                self.adata.write(adata_file)
            else:
                self.adata = sc.read_h5ad(adata_file)

    def inverse_affine_transform(self, x_pixel, y_pixel):
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
        centroid = self.cell_boundaries.centroid
        xenium_x_um, xenium_y_um = centroid.x, centroid.y
        xenium_x_px, xenium_y_px = xenium_x_um / self.pixel_size, xenium_y_um / self.pixel_size
        he_x, he_y = inverse_affine_transform(xenium_x_px, xenium_y_px)
        he_x, he_y = round(he_x), round(he_y)
        half = self.crop_size // 2
        x_min = int(he_x - half)
        x_max = int(he_x + half)
        y_min = int(he_y - half)
        y_max = int(he_y + half)

        image_path = self.processed_path + f'he{self.crop_size}/'
        os.makedirs(image_path, exist_ok=True)
        if 0 <= x_min and x_max < self.image_width and 0 <= y_min and y_max < self.image_height:
            cropped_image = self.he_image_array[:, y_min:y_max, x_min:x_max]
            cropped_image = cropped_image.transpose(1, 2, 0) # (c, y, x) to (y, x, c)
            cropped_image = Image.fromarray(cropped_image)
            cropped_image.save(image_path + f"{cell_id}.png")
        
    def cell_area_filter(self):
        bounds = self.cell_boundaries.bounds
        len_raw = len(bounds)
        bounds = bounds / self.pixel_size
        bounds['width'] = bounds.apply(lambda row: row['maxx'] - row['minx'], axis=1)
        bounds['height'] = bounds.apply(lambda row: row['maxy'] - row['miny'], axis=1)
        bounds = bounds.merge(self.adata.obs['Cell_type'], how='left', left_index=True, right_index=True)
        bounds_melted = bounds.melt(id_vars='Cell_type', value_vars=['width', 'height'], var_name='Length', value_name='Value')


        fig = plt.figure(figsize=(4, 4))
        sns.violinplot(bounds_melted, x='Value', y='Cell_type', hue='Length', split=True)
        fig.tight_layout()
        fig.savefig(f"/data0/crp/results/violinplot_{self.platform}_{self.sample}.png", bbox_inches="tight")
        plt.close(fig)
        
        self.filtered_cell_ids = bounds[
            (self.lower_bound <= bounds['width']) & 
            (bounds['width'] < self.upper_bound) &
            (self.lower_bound <= bounds['height']) & 
            (bounds['height'] < self.upper_bound)
        ].index
        print(f"{len(self.filtered_cell_ids)} cells are selected from {len_raw} cells.")

    def crop_cells(self):
        self.cell_ids = set(self.annotated_cell_ids) & set(self.filtered_cell_ids)
        print(f'{len(self.cell_ids)} cells are prepared for model')
        pool = Pool(n_cores)
        expression_dfs = pool.map(self._crop_he_image, tqdm(self.cell_ids))
        pool.close()
        pool.join()

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

    preprocess = Preprocess(args.platform, args.sample)
    preprocess.annotation()
    preprocess.cell_area_filter()
    preprocess.crop_cells()
