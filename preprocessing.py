import os
from pathlib import Path
import shutil
import time
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import sparse
import scanpy as sc
import matplotlib.pyplot as plt
import spatialdata_plot
import spatialdata as sd
from spatialdata_io import xenium

'''
Global configuration
'''
N_CORES = mp.cpu_count()
PIXEL_SIZE = 0.2125  # micrometers per pixel
UPPER_THRESHOLD = 200
ORGAN = '10x_5k_lung'
AFFINE_MATRIX_PATH = Path("/data0/xenium_prime/Human_Lung_Cancer/Xenium_Prime_Human_Lung_Cancer_FFPE_he_imagealignment.csv")
DATA_DIR = Path("data")
ZARR_DIR = Path("/data0/xenium_prime/Human_Lung_Cancer/data.zarr")

'''
Parse and save Xenium data once
'''
def prepare_spatial_data(data_dir: Path, zarr_dir: Path):
    zarr_dir.parent.mkdir(parents=True, exist_ok=True)
    if not zarr_dir.exists():
        print("Parsing Xenium data...", end=" ")
        sdata = xenium(
            str(data_dir), n_jobs=N_CORES, cell_boundaries=True,
            nucleus_boundaries=True, morphology_focus=True, cells_as_circles=True
        )
        print("done")
        print("Saving to Zarr...", end=" ")
        if zarr_dir.exists(): 
            shutil.rmtree(zarr_dir)
        sdata.write(str(zarr_dir))
        print("done")
    return sd.SpatialData.read(str(zarr_dir))

'''
Geometric utilities
'''
def load_affine_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).values

def apply_affine(x_px: float, y_px: float, affine: np.ndarray) -> tuple[float, float]:
    vec = np.array([x_px, y_px, 1.0])
    x, y, _ = affine.dot(vec)
    return x, y

'''
Image cropping
'''
def crop_he_image(
        cell_id: str, sdata: sd.SpatialData, affine: np.ndarray,
        he_image: np.ndarray, crop_size: int
    ) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    geom = sdata.shapes['cell_boundaries'].loc[cell_id, 'geometry']
    cx_um, cy_um = geom.centroid.x, geom.centroid.y
    cx_px, cy_px = cx_um / PIXEL_SIZE, cy_um / PIXEL_SIZE
    hx, hy = apply_affine(cx_px, cy_px, np.linalg.inv(affine))
    half = crop_size // 2
    x0, y0 = max(0,int(hx-half)), max(0,int(hy-half))
    x1, y1 = min(he_image.shape[2], x0+crop_size), min(he_image.shape[1], y0+crop_size)
    return he_image[:, y0:y1, x0:x1], (x0,x1,y0,y1)

'''
Selection by the area of cells
'''
def filter_cells(adata_obs: pd.DataFrame) -> tuple[pd.Index,int]:
    adata_obs.index = adata_obs['cell_id']
    low = np.percentile(adata_obs['cell_area'], 10)
    high = UPPER_THRESHOLD
    adata_obs = adata_obs[(low <= adata_obs['cell_area']) & (adata_obs['cell_area'] < high)].copy()
    # determine crop_size by largest area cell
    cell_id = adata_obs.loc[adata_obs['cell_id'] == adata_obs['cell_area'].idxmax(), 'cell_id']
    bounds = sdata.shapes['cell_boundaries'].loc[cell_id,'geometry'].bounds
    size = max(
        bounds.maxx.values[0] - bounds.minx.values[0], 
        bounds.maxy.values[0] - bounds.miny.values[0]
    )
    print(f'The length of one side of a circumscribed square is {size}.')
    return adata_obs['cell_id'], size

'''
Helper for multiprocessing
'''
def _process_single_cell(cell_id, sdata, affine, he_image, crop_size):
    image, _ = crop_he_image(cell_id, sdata, affine, he_image, crop_size)
    if image.size == 0: 
        return None
    obs = sdata.tables['table'].obs
    idx = obs.index[obs['cell_id']==cell_id]
    if len(idx) == 0: 
        return None
    row = obs.loc[idx[0]]
    expression = sdata.tables['table'].X[idx[0]].toarray().flatten()
    df = pd.DataFrame({'gene':sdata.tables['table'].var.index,'expression':exprseeion})
    df['cell_id']=cell_id
    return df

'''
Main processing
'''
def process_cells(sdata: sd.SpatialData, annotation: pd.DataFrame):
    he_image = sdata.images['he_image']['scale0']['image'].values
    adata = sdata.tables['table']
    affine = load_affine_matrix(AFFINE_MATRIX_PATH)
    cells, crop_size = filter_cells(adata.obs)
    args = [(cid, sdata, affine, he_image, crop_size) for cid in cells]

    with Pool(N_CORES) as pool:
        dfs = pool.starmap(_process_single_cell, args)
    dfs = [df for df in dfs if df is not None]
    expr = pd.concat(dfs)
    expr_mat = expr.pivot(index='cell_id',columns='gene',values='expression').fillna(0)
    # save matrices and AnnData
    adata = sc.AnnData(
        X=sparse.csr_matrix(expr_mat.values),
        obs=annotation.set_index('cell_id'),
        var=pd.DataFrame(index=expr_mat.columns),
    )
    out_dir = Path(f"/data0/paired_dataset/{ORGAN}")
    out_dir.mkdir(parents=True, exist_ok=True)
    adata.write(out_dir/f"expression_he{UPPER_THRESHOLD}.h5ad")
    expr_mat.T.pipe(lambda df: df.to_csv(out_dir/f"gene_by_cell_expr_nonzero.csv"))
    return adata

if __name__=='__main__':
    sdata = prepare_spatial_data(DATA_DIR, ZARR_DIR)
    annotation = pd.read_csv('/data0/crp/annotation/merged_outputs.csv')
    annotation['cell_id']=annotation['cell_id'].astype(str)
    adata = process_cells(sdata, annotation)
    print("Pipeline completed. AnnData saved.")
