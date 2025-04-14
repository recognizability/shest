import os
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch

import seaborn as sns
import matplotlib.pyplot as plt

from utils import PairedDataset
from model import Reconstruction, Classification

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

cell_types_cz = {
    'Tumor_cell_LUAD': [
        'Tumor cells LUAD',
        'Tumor cells LUAD EMT',
        'Tumor cells LUAD MSLN',
        'Tumor cells LUAD NE',
        'Tumor cells LUAD mitotic'
    ],
    'Stromal_cell': [
        'stromal dividing',
        'Fibroblast adventitial',
        'Fibroblast alveolar',
        'Fibroblast peribronchial'
    ],
    'Pericyte':[
        'Pericyte',
    ],
    'Endothelial_cell': [
        'Endothelial cell arterial',
        'Endothelial cell capillary',
        'Endothelial cell lymphatic',
        'Endothelial cell venous',
    ],
    'Lymphocyte': [
         'B cell',
         'B cell dividing',
         'Plasma cell',
         'Plasma cell dividing',
         'T cell regulatory',
         'T cell CD4',
         'T cell CD4 dividing',
         'T cell CD8 activated',
         'T cell CD8 dividing',
         'T cell CD8 effector memory',
         'T cell CD8 naive',
         'T cell CD8 terminally exhausted',
         'T cell NK-like',
         'NK cell',
         'NK cell dividing'
    ]
}

parser = argparse.ArgumentParser(description="Sample information and hyperparameters")
parser.add_argument("--sample", type=str, default="10x_5k_lung", help="Sample name consisting of platform, panel size, and organ")
parser.add_argument("--he", type=str, default="he200", help="Side length of H&E image")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size of data loader")
parser.add_argument("--epochs", type=int, default=20, help="Number of epochs in training")
parser.add_argument("--lr", type=int, default=0.01, help="Learning rate of optimizer")
args = parser.parse_args()

n_cores = max(os.cpu_count()-2, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Dataset
'''
directory = '/data0/paired_dataset/'
paired_dataset = PairedDataset(n_cores, directory, args.sample, args.he, cell_types_cz)
adata = paired_dataset.cell_select()
paired_dataset.draw_umaps_expression()
palette_he = paired_dataset.palette_he
train_loader, test_loader = paired_dataset.loaders(seed, args.batch_size)

'''
Reconstruction
'''
reconstruction = Reconstruction(seed, adata, args.sample, args.he)
reconstruction.train(train_loader, args.epochs, args.lr, force=False)
reconstruction.evaluate(test_loader)
reconstruction.draw_umaps_embedding(palette_he)
reconstruction.draw_heatmap(cell_types_cz, palette_he)

'''
Classification
'''
for cell_type in ['cell_type_common', 'cell_subtype_st']:
    classification = Classification(adata, args.sample, args.he, cell_types_cz, cell_type)
    classification.train(train_loader, args.epochs, args.lr, force=True)
    classification.evaluate(test_loader)
