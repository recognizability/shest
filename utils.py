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
from sklearn.preprocessing import LabelEncoder

import tacco as tc
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from config import n_cores, seed, set_seed, generator, device

sc.settings.n_jobs = n_cores

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocessing(adata):
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :].copy()

    return adata

class PairedDataset():
    def __init__(self, directory, platform, source, sample, he, cell_type, cell_types, angles):
        self.directory = directory
        self.platform = platform
        self.source = source
        self.sample = sample
        self.he = he
        self.cell_type = cell_type
        self.cell_types = cell_types
        self.angles = angles

        self.palette_type = dict(zip(
            self.cell_types.keys(), 
            sns.color_palette('blend:red,orange,green,blue', n_colors=len(cell_types.keys())).as_hex()
        ))
        self.cell_subtypes = sum(cell_types.values(), [])
        self.palette_subtype = dict(zip(
            self.cell_subtypes,
            sns.color_palette('blend:red,orange,green,blue', n_colors=len(self.cell_subtypes)).as_hex()
        ))

        base_dir = os.path.join(self.directory, self.platform, self.source, self.sample)
        image_dir = os.path.join(base_dir, self.he, self.cell_type)
        image_files = glob(os.path.join(image_dir, '*.png'))
        image_ids = [cell.split('/')[-1].split('.')[0] for cell in image_files]
        print(len(image_ids), "images of the cells are prepared.")

        self.label_encoder = LabelEncoder()
        if self.cell_type == 'Cell_type' or self.cell_type == 'Cell_type_ST' or self.cell_type == 'Cell_type_HE':
            self.parameters = list(self.cell_types.keys())
        elif self.cell_type == 'Cell_subtype_ST':
            self.parameters =  self.cell_subtypes
        self.label_encoder.fit(self.parameters)
        self.label_encoder.classes_ = np.array(self.parameters)
        self.classes = self.label_encoder.classes_
    
        print('Expression profile and their cell types loading ... ', end='')
        adata_file = os.path.join(base_dir, f'annotation/adata_{self.he}.h5ad')
        self.adata_raw = sc.read_h5ad(adata_file)
        print(self.adata_raw.shape)
        type_ids = self.adata_raw.obs[self.cell_type].dropna().index.tolist()
        print(len(type_ids), "of annotated cells are loaded.")

        self.cell_ids = sorted(list(set(image_ids) & set(type_ids)))
        print('Common', len(self.cell_ids), "cells are selected.")

        self.image_files = pd.Series(self.cell_ids, index=self.cell_ids).apply(
            lambda file: os.path.join(image_dir, f"{file}.png")
        ).to_dict()

        self.adata = self.adata_raw[self.cell_ids, :].copy()
        print(self.adata.obs[self.cell_type].value_counts())
        self.cell_type_encoded = self.cell_type + '_encoded'
        self.adata.obs[self.cell_type_encoded] = self.label_encoder.transform(self.adata.obs[self.cell_type])
        self.var_names = self.adata.var_names.tolist()

        cell_indices = self.adata.obs_names.get_indexer(self.cell_ids)
        self.expressions = self.adata.X[cell_indices].toarray().astype(np.float32)
        self.labels = self.adata.obs.iloc[cell_indices][self.cell_type_encoded].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.cell_ids) * len(self.angles)

    def __getitem__(self, i):
        base_i = i // len(self.angles)
        angle_i = i % len(self.angles)

        cell_id = self.cell_ids[base_i]
        angle = self.angles[angle_i]

        image = cv2.imread(self.image_files[cell_id])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transforms.functional.rotate(image, angle)
        image = transform(image)

        expression = torch.from_numpy(self.expressions[base_i])
        label = torch.tensor(self.labels[base_i], dtype=torch.long)

        return cell_id, image, expression, label

    def draw_umaps_expression(self):
        fig, ax = plt.subplots(3, 2, figsize=(7, 9))
        for i, cell_type in enumerate(['Cell_type_ST', 'Cell_type_HE', 'Cell_type']):
            cell_types = self.adata_raw.obs[cell_type].value_counts().index.tolist()
            sns.barplot(
               pd.DataFrame(self.adata_raw.obs.groupby(cell_type).apply(len, include_groups=False), columns=['']).reindex(self.cell_types.keys()).T,
               orient = 'h',
               palette = self.palette_type,
               ax=ax[i][0]
            )
            sc.pl.umap(self.adata_raw, color=cell_type, palette=self.palette_type, ax=ax[i][1], show=False, legend_loc=None)

        fig.tight_layout()
        fig.savefig(f"/data0/crp/results/umaps_expression_{self.platform}_{self.source}_{self.sample}_{self.he}.png", bbox_inches="tight")
        plt.close()

    def get_dataloaders(self, batch_size, split=0.8):
        total = len(self)
        train_size = int(split * total)
        test_size = total - train_size
        train_dataset, test_dateset = random_split(self, [train_size, test_size], generator=generator)
        print(f"Train size: {len(train_dataset)}, Test size: {len(test_dateset)}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cores, pin_memory=True)
        test_loader = DataLoader(test_dateset, batch_size=batch_size, shuffle=False, num_workers=n_cores, pin_memory=True)
        return self.var_names, self.classes, train_loader, test_loader
