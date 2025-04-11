import os
os.environ["PARAMETRICUMAP"] = "0"
os.environ["UMAP_DISABLE_PARAMETRIC"] = "True"

import gc
from glob import glob
from collections import Counter
from tqdm import tqdm
import umap

import numpy as np
import pandas as pd
import scanpy as sc
import anndata

import tacco as tc
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class HEDataset(Dataset):
    def __init__(self, cell_ids, directory, sample, he, transform=transform):
        self.cell_ids = cell_ids
        self.directory = directory
        self.transform = transform
        self.sample = sample
        self.he = he

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        image_path = os.path.join(self.directory, self.sample, self.he, f'{cell_id}.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return cell_id, image

def log_normalize(adata, target_sum=1e4):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata

def preprocessing(adata):
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :].copy()

    return adata

def single_cell_reference(organ):
    if organ == 'lung':
        print("Loading LuCA single cell reference ... ", end='')
        luca = sc.read_h5ad('/data0/cz_sc_reference/dd538ee7-f5e4-49e9-9f1e-2a1ea5246cf4.h5ad')
        luca = luca[luca.obs['cell_type_tumor']!='ROS1+ healthy epithelial'].copy()
        luca.index = luca.var.feature_name
        luca.var.index = luca.var.feature_name
        print(luca.shape)
        return luca

def cell_type_annotation(adata, cell_types, sample, he, n_cores, cell_subtype='cell_type_tumor'):
    print('Annotating types of the cells ... ')
    
    annotation_file = f"/data0/crp/annotation/cell_subtype_{sample}_{he}.h5ad"
    
    if not os.path.isfile(annotation_file):
        ref = single_cell_reference(sample.split('_')[-1])

        print('Preprocessing adata ... ', end='')
        adata = preprocessing(adata)
        print(adata.shape)
    
        adata.obs['Cell_subtype'] = tc.tl.annotate(adata, ref, annotation_key=cell_subtype, assume_valid_counts=True).T.idxmax()
        adata.obs['Cell_type'] = adata.obs['Cell_subtype'].map({cell: group for group, cells in cell_types.items() for cell in cells}).astype('category')

        adata.write(annotation_file)
        
    else:
        adata = sc.read_h5ad(annotation_file)

    print(len(adata.obs['Cell_subtype'].unique()), 'subtypes are annotated.')
    
    print('Categorize the cell subtypes into cell types ...')
    adata.obs['cell_subtype_st'] = adata.obs['Cell_subtype'].map({cell: cell for group, cells in cell_types.items() for cell in cells}).astype('category')
    adata.obs['cell_type_st'] = adata.obs['Cell_subtype'].map({cell: group for group, cells in cell_types.items() for cell in cells}).astype('category')
    print(len(adata.obs['cell_type_st'].unique()), 'types are annotated.')

    return adata

class PairedDataset():
    def __init__(self, n_cores, directory, sample, he, cell_types_cz):
        self.n_cores = n_cores
        self.directory = directory
        self.sample = sample
        self.he = he
        self.cell_types_cz = cell_types_cz
        self.palette_he = dict(zip(
            cell_types_cz.keys(),
            sns.color_palette('blend:red,black,green', n_colors=len(cell_types_cz.keys())).as_hex()
        ))
        self.cell_subtypes_st = sum(cell_types_cz.values(), [])

        self.palette_st = dict(zip(
            self.cell_subtypes_st,
            sns.color_palette('blend:red,black,green', n_colors=len(self.cell_subtypes_st)).as_hex()
        ))

        cells_path = os.path.join(directory, sample, he, '*')
        self.cell_ids = [cell.split('/')[-1].split('.')[0] for cell in glob(cells_path)]
        dataset = HEDataset(self.cell_ids, directory, sample, he)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.n_cores, pin_memory=True)

        self.adata = None
        self.common_ids = None
        self.dataloader_selected = None

    def cell_select(self, cell_type_he='group', force=False):
        annotation_file = f"/data0/crp/annotation/cell_type_{self.sample}_{self.he}.h5ad"
        
        if not os.path.isfile(annotation_file) or force:
            print('Expression profile and their cell types loading ... ', end='')
            expression_file = os.path.join(self.directory, self.sample, f'expression_{self.he}.h5ad')
            self.adata = sc.read_h5ad(expression_file)
            self.adata.obs['cell_type_he'] = self.adata.obs[cell_type_he]
            print(self.adata.shape)
            
            self.adata = cell_type_annotation(self.adata, cell_types=self.cell_types_cz, sample=self.sample, he=self.he, n_cores=self.n_cores)
            self.adata = log_normalize(self.adata)
            sc.pp.neighbors(self.adata)
            sc.tl.umap(self.adata)
        
            self.adata.obs['cell_type_common'] = np.nan
            condition = self.adata.obs['cell_type_st'].astype('str') == self.adata.obs['cell_type_he'].astype('str')
            self.adata.obs.loc[condition, 'cell_type_common'] = self.adata.obs.loc[condition, 'cell_type_he']
            self.adata.write(annotation_file)
            
        else:
            self.adata = sc.read_h5ad(annotation_file)
        
        self.common_ids = self.adata[self.adata.obs['cell_type_common'].notna(), :].obs.index
        print('Common', len(self.common_ids), "cells are selected.")
        selected_indices = [self.cell_ids.index(cell_id) for cell_id in self.cell_ids if cell_id in self.common_ids]
        self.dataloader_selected = DataLoader(Subset(self.dataloader.dataset, selected_indices), batch_size=self.dataloader.batch_size, shuffle=False)

        return self.adata

    def draw_umaps_expression(self):
        fig, ax = plt.subplots(4, 2, figsize=(8, 12))

        sns.barplot(
           pd.DataFrame(self.adata.obs.groupby(['cell_subtype_st']).apply(len, include_groups=False), columns=['']).reindex(self.cell_subtypes_st).T,
           orient = 'h',
           palette = self.palette_st,
           ax=ax[0][0]
        )
        sc.pl.umap(self.adata, color='cell_subtype_st', palette=self.palette_st, ax=ax[0][1], show=False, legend_loc=None)

        sns.barplot(
           pd.DataFrame(self.adata.obs.groupby(['cell_type_st']).apply(len, include_groups=False), columns=['']).reindex(self.cell_types_cz.keys()).T,
           orient = 'h',
           palette = self.palette_he,
           ax=ax[1][0]
        )
        sc.pl.umap(self.adata, color='cell_type_st', palette=self.palette_he, ax=ax[1][1], show=False, legend_loc=None)

        sns.barplot(
           pd.DataFrame(self.adata.obs.groupby(['cell_type_he']).apply(len, include_groups=False), columns=['']).reindex(self.cell_types_cz.keys()).T,
           orient = 'h',
           palette = self.palette_he,
           ax=ax[2][0]
        )
        sc.pl.umap(self.adata, color='cell_type_he', palette=self.palette_he, ax=ax[2][1], show=False, legend_loc=None)

        sns.barplot(
           pd.DataFrame(self.adata.obs.groupby(['cell_type_common']).apply(len, include_groups=False), columns=['']).reindex(self.cell_types_cz.keys()).T,
           orient = 'h',
           palette = self.palette_he,
           ax=ax[3][0]
        )
        sc.pl.umap(self.adata, color='cell_type_common', palette=self.palette_he, ax=ax[3][1], show=False, legend_loc=None)

        fig.tight_layout()
        fig.savefig(f"/data0/crp/results/umaps_expression_{self.sample}_{self.he}.png", bbox_inches="tight")
        plt.close()

    def loaders(self, seed, batch_size):
        train_size = int(0.8 * len(self.common_ids))
        test_size = len(self.common_ids) - train_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, test_dataset = random_split(self.dataloader_selected.dataset, [train_size, test_size], generator=generator)
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.n_cores, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.n_cores, pin_memory=True) 
        return train_loader, test_loader
