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

from config import n_cores, seed, set_seed

sc.settings.n_jobs = n_cores

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class HEDataset(Dataset):
    def __init__(self, cell_ids, directory, platform, sample, he, angles=[0]):
        self.cell_ids = cell_ids
        self.directory = directory
        self.platform = platform
        self.sample = sample
        self.he = he
        self.angles = angles

    def __len__(self):
        return len(self.cell_ids) * len(self.angles)

    def __getitem__(self, i):
        base_i = i // len(self.angles)
        angle_i = i % len(self.angles)

        cell_id = self.cell_ids[base_i]
        angle = self.angles[angle_i]

        image_path = os.path.join(self.directory, self.platform, self.sample, self.he, f'{cell_id}.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transforms.functional.rotate(image, angle)
        image = transform(image)
        return cell_id, image

def preprocessing(adata):
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :].copy()

    return adata

class PairedDataset():
    def __init__(self, directory, platform, sample, he, cell_types):
        self.directory = directory
        self.platform = platform
        self.sample = sample
        self.he = he
        self.cell_types = cell_types
        self.palette_type = dict(zip(
            self.cell_types.keys(), 
            sns.color_palette('blend:red,orange,green,blue', n_colors=len(cell_types.keys())).as_hex()
        ))
        self.cell_subtypes = sum(cell_types.values(), [])
        self.palette_subtype = dict(zip(
            self.cell_subtypes,
            sns.color_palette('blend:red,orange,green,blue', n_colors=len(self.cell_subtypes)).as_hex()
        ))

        cells_path = os.path.join(directory, platform, sample, he, '*.png')
        self.image_ids = [cell.split('/')[-1].split('.')[0] for cell in glob(cells_path)]
        self.dataloader = None

        self.adata = None
        self.type_ids = None
        self.common_ids = None

    def cell_select(self, force=False):
        print('Expression profile and their cell types loading ... ', end='')
        adata_file = f'{self.directory}{self.platform}/{self.sample}/annotation/adata_{self.he}.h5ad'
        self.adata = sc.read_h5ad(adata_file)
        print(self.adata.shape)
        
        self.type_ids = self.adata[self.adata.obs['Cell_type'].notna(), :].obs.index
        self.common_ids = list(set(self.image_ids) & set(self.type_ids))
        self.common_ids.sort()
        print('Common', len(self.common_ids), "cells are selected.")

        return self.adata

    def draw_umaps_expression(self):
        fig, ax = plt.subplots(3, 2, figsize=(7, 9))

        sns.barplot(
           pd.DataFrame(self.adata.obs.groupby(['Cell_type_ST']).apply(len, include_groups=False), columns=['']).reindex(self.cell_types.keys()).T,
           orient = 'h',
           palette = self.palette_type,
           ax=ax[0][0]
        )
        sc.pl.umap(self.adata, color='Cell_type_ST', palette=self.palette_type, ax=ax[0][1], show=False, legend_loc=None)

        sns.barplot(
           pd.DataFrame(self.adata.obs.groupby(['Cell_type_HE']).apply(len, include_groups=False), columns=['']).reindex(self.cell_types.keys()).T,
           orient = 'h',
           palette = self.palette_type,
           ax=ax[1][0]
        )
        sc.pl.umap(self.adata, color='Cell_type_HE', palette=self.palette_type, ax=ax[1][1], show=False, legend_loc=None)

        sns.barplot(
           pd.DataFrame(self.adata.obs.groupby(['Cell_type']).apply(len, include_groups=False), columns=['']).reindex(self.cell_types.keys()).T,
           orient = 'h',
           palette = self.palette_type,
           ax=ax[2][0]
        )
        sc.pl.umap(self.adata, color='Cell_type', palette=self.palette_type, ax=ax[2][1], show=False, legend_loc=None)

        fig.tight_layout()
        fig.savefig(f"/data0/crp/results/umaps_expression_{self.platform}_{self.sample}_{self.he}.png", bbox_inches="tight")
        plt.close()

    def loaders(self, batch_size):
        angles = [
            0,
#            90,
#            180,
#            270,
        ]
        full_dataset = HEDataset(self.common_ids, self.directory, self.platform, self.sample, self.he, angles=angles)
        train_size = int(0.8 * len(self.common_ids))
        test_size = len(self.common_ids) - train_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, test_dataset = random_split(full_dataset, [train_size*len(angles), test_size*len(angles)], generator=generator)
        print(f'The training dataset size is {len(train_dataset)}, and the test dataset size is {len(test_dataset)}.')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cores, pin_memory=True, generator=generator)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cores, pin_memory=True) 
        return train_loader, test_loader
