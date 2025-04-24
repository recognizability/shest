# Xenium Data Preparation and Processing Pipeline

import os
from pathlib import Path
import subprocess
import shutil
import time
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse
import scanpy as sc
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import spatialdata_plot
import spatialdata as sd
from spatialdata_io import xenium

# Global configuration
NUM_CORES = mp.cpu_count()
PIXEL_SIZE = 0.2125  # micrometers per pixel
UPPER_THRESHOLD = 200
ORGAN = '10x_5k_lung'
AFFINE_MATRIX_PATH = Path("./xenium_prime_link/Human_Lung_Cancer/Xenium_Prime_Human_Lung_Cancer_FFPE_he_imagealignment.csv")
DATA_DIR = Path("data")
ZARR_DIR = Path("xenium_link/Human_Lung_Cancer/data.zarr")

# Parse and save Xenium data once

def prepare_spatial_data(data_dir: Path, zarr_dir: Path):
    zarr_dir.parent.mkdir(parents=True, exist_ok=True)
    if not zarr_dir.exists():
        print("Parsing Xenium data...", end=" ")
        sdata = xenium(str(data_dir), n_jobs=NUM_CORES, cell_boundaries=True,
                        nucleus_boundaries=True, morphology_focus=True, cells_as_circles=True)
        print("done")
        print("Saving to Zarr...", end=" ")
        if zarr_dir.exists(): shutil.rmtree(zarr_dir)
        sdata.write(str(zarr_dir))
        print("done")
    return sd.SpatialData.read(str(zarr_dir))

# Geometric utilities

def load_affine_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).values

def apply_affine(x_px: float, y_px: float, affine: np.ndarray) -> tuple[float, float]:
    vec = np.array([x_px, y_px, 1.0])
    x, y, _ = affine.dot(vec)
    return x, y

# Image cropping

def crop_he_image(cell_id: str, sdata: sd.SpatialData, affine: np.ndarray,
                  he_img: np.ndarray, crop_size: int) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    geom = sdata.shapes['cell_boundaries'].loc[cell_id, 'geometry']
    cx_um, cy_um = geom.centroid.x, geom.centroid.y
    cx_px, cy_px = cx_um / PIXEL_SIZE, cy_um / PIXEL_SIZE
    hx, hy = apply_affine(cx_px, cy_px, np.linalg.inv(affine))
    half = crop_size // 2
    x0, y0 = max(0,int(hx-half)), max(0,int(hy-half))
    x1, y1 = min(he_img.shape[2], x0+crop_size), min(he_img.shape[1], y0+crop_size)
    return he_img[:, y0:y1, x0:x1], (x0,x1,y0,y1)

# Postprocessing

def filter_cells(trans_obs: pd.DataFrame) -> tuple[pd.Index,int]:
    areas = trans_obs['cell_area']
    low, high = np.percentile(areas,10), UPPER_THRESHOLD
    filt = trans_obs[(areas>=low)&(areas<high)]
    # determine crop_size by largest area cell
    cid = filt.loc[areas.idxmax(),'cell_id']
    poly = sdata.shapes['cell_boundaries'].loc[cid,'geometry']
    rect = poly.minimum_rotated_rectangle.exterior.coords[:-1]
    edges = np.linalg.norm(np.diff(rect,axis=0),axis=1)
    size = int(max(edges)/PIXEL_SIZE)
    size += size%2
    return filt['cell_id'], size

# Main processing

def process_cells(sdata: sd.SpatialData, annotation: pd.DataFrame):
    he_img = sdata.images['he_image']['scale0']['image'].values
    transcript = sdata.tables['table']
    affine = load_affine_matrix(AFFINE_MATRIX_PATH)
    cells, crop_size = filter_cells(transcript.obs)
    args = [(cid, sdata, affine, he_img, crop_size) for cid in cells]
    with Pool(NUM_CORES) as pool:
        dfs = pool.starmap(_process_single_cell, args)
    dfs = [df for df in dfs if df is not None]
    expr = pd.concat(dfs)
    expr_mat = expr.pivot(index='cell_id',columns='gene',values='expression').fillna(0)
    # save matrices and AnnData
    adata = sc.AnnData(X=sparse.csr_matrix(expr_mat.values),
                       obs=annotation.set_index('cell_id'),
                       var=pd.DataFrame(index=expr_mat.columns))
    out_dir = Path(f"/data0/paired_dataset/{ORGAN}")
    out_dir.mkdir(parents=True,exist_ok=True)
    adata.write(out_dir/f"expression_he{UPPER_THRESHOLD}.h5ad")
    expr_mat.T.pipe(lambda df: df.to_csv(out_dir/f"gene_by_cell_expr_nonzero.csv"))
    return adata

# Helper for multiprocessing

def _process_single_cell(cell_id, sdata, affine, he_img, crop_size):
    img, _ = crop_he_image(cell_id, sdata, affine, he_img, crop_size)
    if img.size==0: return None
    obs = sdata.tables['table'].obs
    idx = obs.index[obs['cell_id']==cell_id]
    if len(idx)==0: return None
    row = obs.loc[idx[0]]
    expr = sdata.tables['table'].X[idx[0]].toarray().flatten()
    df = pd.DataFrame({'gene':sdata.tables['table'].var.index,'expression':expr})
    df['cell_id']=cell_id
    return df

if __name__=='__main__':
    sdata = prepare_spatial_data(DATA_DIR, ZARR_DIR)
    annotation = pd.read_csv('annotation/merged_output.csv')
    annotation['cell_id']=annotation['cell_id'].astype(str)
    adata = process_cells(sdata, annotation)
    print("Pipeline completed. AnnData saved.")
