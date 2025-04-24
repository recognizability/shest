#!/usr/bin/env python
# coding: utf-8

# # Data_preperation

# In[ ]:


from pathlib import Path
import shutil
from spatialdata_io import xenium
import spatialdata as sd

path = Path().resolve()

#set paths
path /= "xenium_link/Human_Lung_Cancer"
path.mkdir(parents=True, exist_ok=True) 


path_read = path / "data"  
path_write = path / "data.zarr" 

sdata = xenium(
    path=str(path_read),
    n_jobs=8,
    cell_boundaries=True,
    nucleus_boundaries=True,
    morphology_focus=True,
    cells_as_circles=True,
)

if path_write.exists():
    shutil.rmtree(path_write)  
sdata.write(path_write) 


sdata = sd.SpatialData.read(str(path_write))



# # Packages

# In[1]:


import spatialdata as sd
import matplotlib.pyplot as plt
import spatialdata_plot
import shapely
import numpy as np
import xarray as xr
import skimage.transform
import pandas as pd
from shapely.geometry import Polygon
import geopandas as gpd
from shapely.geometry import Point
import cv2
import os
import time
import multiprocessing as mp
import scanpy as sc
import pandas as pd
from scipy import sparse
from multiprocessing import Pool


# # Global variable

# In[3]:


NUM_CORES = 32
AFFINE_MATRIX_PATH = "./xenium_prime_link/Human_Lung_Cancer/Xenium_Prime_Human_Lung_Cancer_FFPE_he_imagealignment.csv"
PIXEL_SIZE = 0.2125  
UPPER_THERSHOLD = 200
ORGAN ='10x_5k_lung'


# # Functions

# In[4]:


def convert_um_to_px(um_x, um_y, PIXEL_SIZE):
    px_x = um_x / PIXEL_SIZE
    px_y = um_y / PIXEL_SIZE
    return px_x, px_y


# In[5]:


def apply_affine(x_pixel, y_pixel, affine_matrix):
    vec = np.array([[x_pixel], [y_pixel], [1]])  
    x_real, y_real, _ = (affine_matrix @ vec).flatten() 
    return x_real, y_real


# In[6]:


def apply_affine_to_polygon(polygon, affine_matrix):
    coords = np.array(polygon.exterior.coords)
    transformed_coords = np.array([apply_affine(x, y, np.linalg.inv(affine_matrix)) for x, y in coords])
    return Polygon(transformed_coords)


# In[7]:


def load_affine_matrix(AFFINE_MATRIX_PATH):
    affine_matrix = pd.read_csv(AFFINE_MATRIX_PATH, header=None).values
    return affine_matrix


# In[8]:


def convert_polyum_to_px(polygon, PIXEL_SIZE):
    coords = np.array(polygon.exterior.coords)
    coords_px = coords / PIXEL_SIZE  # μm → pixel
    return Polygon(coords_px)


# In[9]:


def adjust_polygon_for_crop(polygon, crop_x_min, crop_y_min):
    coords = np.array(polygon.exterior.coords)
    adjusted_coords = coords - [crop_x_min, crop_y_min]
    return Polygon(adjusted_coords)


# In[10]:


def crop_he_image(cell_id, PIXEL_SIZE, affine_matrix, he_image_array, image_width, image_height, crop_size):
    """
    
    Parameters:
        - cell_id (str): Cell ID
        - PIXEL_SIZE (float): Xenium pixel size (μm/px)
        - affine_matrix (np.array): Affine matrix
        - he_image_array (np.array): H&E image array (C, Y, X)
        - image_width (int): H&E image width
        - image_height (int): H&E image hight
        - crop_size (int): crop size
    
    Returns:
        - cropped_image (np.array): cropped image (C, Y_crop, X_crop)
        - crop_x_min, crop_x_max, crop_y_min, crop_y_max (int): 
    """



    cell_boundaries = sdata.shapes["cell_boundaries"]


    cell_shape = cell_boundaries.loc[cell_id, "geometry"]
    centroid = cell_shape.centroid

    xenium_x_um, xenium_y_um = centroid.x, centroid.y
    xenium_x_px, xenium_y_px = xenium_x_um / PIXEL_SIZE, xenium_y_um / PIXEL_SIZE

    he_x, he_y = apply_affine(xenium_x_px, xenium_y_px, np.linalg.inv(affine_matrix))


    crop_x_min = max(0, int(he_x - crop_size // 2))
    crop_x_max = min(image_width, int(he_x + crop_size // 2))
    crop_y_min = max(0, int(he_y - crop_size // 2))
    crop_y_max = min(image_height, int(he_y + crop_size // 2))

    cropped_image = he_image_array[:, crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    return cropped_image, crop_x_min, crop_x_max, crop_y_min, crop_y_max


# In[11]:


def get_obb_dimensions(polygon):
    rotated_bbox = polygon.minimum_rotated_rectangle 

    coords = np.array(rotated_bbox.exterior.coords[:-1]) 

    edge_lengths = np.linalg.norm(coords - np.roll(coords, shift=1, axis=0), axis=1)
    width, height = np.sort(edge_lengths)[-2:]  

    return int(width), int(height)


# In[12]:


def process_cell(cell_id):
    
    # ✅ 1.get cropped image
    cropped_image, crop_x_min, crop_x_max, crop_y_min, crop_y_max = crop_he_image(
        cell_id, PIXEL_SIZE, affine_matrix, he_image_array, image_width, image_height, crop_size
    )

    if cropped_image is None:
        print(f"⚠️ Cell ID {cell_id} image is unable to crop")
        return
        
    # ✅ 2. transcript information of cell
    transcript_df = transcript_data.obs[transcript_data.obs["cell_id"] == cell_id].copy()
    transcript_df["cell_id"] = cell_id  # Cell ID 추가
    transcript_df["cell_id"] = transcript_df["cell_id"].astype(str)  

    # ✅ 3. gene expression of cell
    cell_index = transcript_df.index
    if len(cell_index) == 0:
        print(f"⚠️ Cell ID {cell_id} unable to find transcript data")
        return
    
    cell_index = transcript_data.obs.index.get_loc(cell_index[0])
    gene_expression = transcript_data.X[cell_index, :].toarray().flatten()
    gene_names = transcript_data.var.index.to_list()
    gene_expression_df = pd.DataFrame({"gene": gene_names, "expression": gene_expression})

    # ✅ 4.`gene_expression_df` add `cell_id` 
    gene_expression_df["cell_id"] = str(cell_id)

    # ✅ 5. Annotation info
    cell_annotation_info = annotation_df[annotation_df["cell_id"] == cell_id]
    if cell_annotation_info.empty:
        print(f"⚠️ Cell ID {cell_id}unable to find annotation info")
        return

    # ✅ 6. data merge (Transcript + Annotation + Gene Expression)
    merged_df = transcript_df.merge(cell_annotation_info, on="cell_id", how="left")
    merged_df = merged_df.merge(gene_expression_df, on="cell_id", how="left") 

    # ✅ 7. image save
    
    # ✅ dave directory
    save_path = f"/data0/paired_dataset/{ORGAN}/he{UPPER_THERSHOLD}"
    os.makedirs(save_path, exist_ok=True) 
    
    image_filename = f"{save_path}/{cell_id}.png"
    csv_filename = f"{save_path}cell_{cell_id}_transcripts_with_annotation.csv"
    
    # ✅ 8. cropped image save (Matplotlib)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cropped_image.transpose(1, 2, 0))  # (C, Y, X) → (Y, X, C) 
    #ax.set_title("Cropped H&E Image (Aligned to Xenium Cell)")
    ax.axis("off")  
    
    # ✅ 9. image save
    fig.savefig(image_filename, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    
    return gene_expression_df[["gene", "expression", "cell_id"]]


# In[13]:


def process_all_cells(PIXEL_SIZE, AFFINE_MATRIX_PATH, save_csv=True, save_adata=True):

    # ✅ 1. filtered cell id
    filtered_cell_ids = set(filtered_cells["cell_id"])
    final_cell_set = filtered_cell_ids & set(cell_ids)
    final_cell = list(final_cell_set)

    # ✅ 2. parellel process
    pool = Pool(NUM_CORES)
    start_time = time.time()
    expression_dfs = pool.map(process_cell, final_cell)
    pool.close()
    pool.join()
    end_time = time.time()
    print(f"📦 process time : {end_time - start_time:.4f}s")

    # ✅ 3. None 
    expression_dfs = [df for df in expression_dfs if df is not None]

    # ✅ 4. long-form DataFrame merge
    all_expr_df = pd.concat(expression_dfs, axis=0)

    # ✅ 5. wide-form  → Cell-by-Gene matrix
    expr_matrix = all_expr_df.pivot(index="cell_id", columns="gene", values="expression").fillna(0)

    # ✅ 6. AnnData obj generation
    adata = sc.AnnData(
        X=sparse.csr_matrix(expr_matrix.values),
        obs=pd.DataFrame(index=expr_matrix.index),
        var=pd.DataFrame(index=expr_matrix.columns)
    )

    # ✅ 7. annotation info add
    adata.obs = adata.obs.merge(annotation_df.set_index("cell_id"), left_index=True, right_index=True, how="left")

    # ✅ 8. cell type per group
    group_counts = adata.obs["group"].value_counts()
    print("\n📊 ✅ after filtering :")
 

    # ✅ annotation.csv 
    annotation_path = f"/data0/paired_dataset/{ORGAN}/annotation.csv"
    anno_df = pd.read_csv(annotation_path).set_index("cell_id")

    # ✅ 9. save
    if save_csv:
        
        save_path = f"/data0/xenium/{ORGAN}_cell_info_{UPPER_THERSHOLD}"
        os.makedirs(save_path, exist_ok=True)
        csv_filename_counts = f"{save_path}/filtered_group_counts.csv"
        group_counts.to_csv(csv_filename_counts, header=True)
       

        merged_expr_T = anno_df.join(expr_matrix, how="inner").T

        total_genes = merged_expr_T.shape[0]
        total_cells = merged_expr_T.shape[1]

        nonzero_expr_T = merged_expr_T.loc[(merged_expr_T != 0).any(axis=1)]

        remaining_genes = nonzero_expr_T.shape[0]
        remaining_cells = nonzero_expr_T.shape[1]

        csv_filename_expr_filtered = f"{save_path}/gene_by_cell_expression_matrix_with_annotation_non_zero_filtered.csv"
        nonzero_expr_T.to_csv(csv_filename_expr_filtered)
       
        csv_filename_expr_annot = f"{save_path}/gene_by_cell_expression_matrix_with_annotation.csv"
        merged_expr_T.to_csv(csv_filename_expr_annot)

    if save_adata:
        adata_filename = f"/data0/paired_dataset/{ORGAN}/expression_he{UPPER_THERSHOLD}.h5ad"
        adata.write(adata_filename)

    return adata


# In[14]:


def cell_area_filter(cell_areas, UPPER_THERSHOLD):

    lower_percentile = np.percentile(cell_areas, 10)
    UPPER_THERSHOLD = UPPER_THERSHOLD 
    
    filtered_cells = transcript_data.obs[
        (cell_areas >= lower_percentile) & (cell_areas < UPPER_THERSHOLD)
    ]
    
    max_filtered_area_index = filtered_cells["cell_area"].idxmax()
    max_filtered_cell_id = filtered_cells.loc[max_filtered_area_index, "cell_id"]

    min_filtered_area_index = filtered_cells["cell_area"].idxmin()
    min_filtered_cell_id = filtered_cells.loc[min_filtered_area_index, "cell_id"]
    

    cell_boundaries = sdata.shapes["cell_boundaries"].copy()
    max_area_polygon = cell_boundaries.loc[max_filtered_cell_id, "geometry"]
    obb_width, obb_height = get_obb_dimensions(max_area_polygon)

    obb_width_px = obb_width / PIXEL_SIZE
    obb_height_px = obb_height / PIXEL_SIZE
    crop_size = int(obb_width_px)
    crop_size += crop_size % 2  
    
    
    return filtered_cells, crop_size


# In[15]:


def process_all_cells(PIXEL_SIZE, AFFINE_MATRIX_PATH, save_csv=True, save_adata=True):
    
    filtered_cell_ids = set(filtered_cells["cell_id"])
    final_cell_set = filtered_cell_ids & set(cell_ids)
    final_cell = list(final_cell_set)

    pool = Pool(NUM_CORES)
    start_time = time.time()
    expression_dfs = pool.map(process_cell, final_cell)
    pool.close()
    pool.join()
    end_time = time.time()

    expression_dfs = [df for df in expression_dfs if df is not None]

    all_expr_df = pd.concat(expression_dfs, axis=0)

    expr_matrix = all_expr_df.pivot(index="cell_id", columns="gene", values="expression").fillna(0)

    adata = sc.AnnData(
        X=sparse.csr_matrix(expr_matrix.values),
        obs=pd.DataFrame(index=expr_matrix.index),
        var=pd.DataFrame(index=expr_matrix.columns)
    )

    adata.obs = adata.obs.merge(annotation_df.set_index("cell_id"), left_index=True, right_index=True, how="left")

    group_counts = adata.obs["group"].value_counts()
 
    annotation_path = f"/data0/paired_dataset/{ORGAN}/annotation.csv"
    anno_df = pd.read_csv(annotation_path).set_index("cell_id")
    
    if save_csv:
     
        save_path = f"/data0/xenium/{ORGAN}_cell_info_{UPPER_THERSHOLD}"
        os.makedirs(save_path, exist_ok=True)  
        csv_filename_counts = f"{save_path}/filtered_group_counts.csv"
        group_counts.to_csv(csv_filename_counts, header=True)
        
        
        merged_expr_T = anno_df.join(expr_matrix, how="inner").T
        

        nonzero_expr_T = merged_expr_T.loc[(merged_expr_T != 0).any(axis=1)]
        
        csv_filename_expr_filtered = f"{save_path}/gene_by_cell_expression_matrix_with_annotation_non_zero_filtered.csv"
        nonzero_expr_T.to_csv(csv_filename_expr_filtered)


        csv_filename_expr_annot = f"{save_path}/gene_by_cell_expression_matrix_with_annotation.csv"
        merged_expr_T.to_csv(csv_filename_expr_annot)
    

    if save_adata:
        adata_filename = f"/data0/paired_dataset/{ORGAN}/expression_he{UPPER_THERSHOLD}.h5ad"
        adata.write(adata_filename)

    return adata


# # Xenium data -LUNG

# In[16]:


xenium_path = "./xenium_prime_link/Human_Lung_Cancer/data.zarr"


# In[17]:


sdata = sd.read_zarr(xenium_path)


# # annotation df generation

# In[18]:


annotation_file = "annotation/merged_output.csv"  
annotation_df = pd.read_csv(annotation_file)


# In[19]:


# cell_id as index
annotation_clean = annotation_df.set_index("cell_id")[["group"]]


# In[20]:


dir_path = f"/data0/paired_dataset/{ORGAN}"
file_path = os.path.join(dir_path, "annotation.csv")

os.makedirs(dir_path, exist_ok=True)

annotation_clean.to_csv(file_path)


# # Cell Extraction -LUNG

# In[21]:


annotation_df["cell_id"] = annotation_df["cell_id"].astype(str)


# In[22]:


transcript_data = sdata.tables["table"]


# In[23]:


he_image = sdata.images["he_image"]["scale0"]
he_image_array = he_image["image"].values 
image_channels, image_height, image_width = he_image_array.shape  # (C, Y, X)


# In[24]:


cell_ids = annotation_df["cell_id"].unique()


# In[25]:


cell_areas = transcript_data.obs["cell_area"]


# In[26]:


affine_matrix = load_affine_matrix(AFFINE_MATRIX_PATH)


# In[27]:


filtered_cells, crop_size = cell_area_filter(transcript_data.obs["cell_area"], UPPER_THERSHOLD)


# In[28]:


process_all_cells(PIXEL_SIZE, AFFINE_MATRIX_PATH)


# In[ ]:




