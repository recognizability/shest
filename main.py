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

parser = argparse.ArgumentParser(
    description="Sample information and hyperparameters",                             
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--sample", type=str, default="10x_5k_lung", help="Sample name consisting of platform, panel size, and organ")
parser.add_argument("--he", type=str, default="he200", help="Side length of H&E image")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size of data loader")
parser.add_argument("--epochs", type=int, default=20, help="Number of epochs in training")
parser.add_argument("--lr", type=int, default=0.01, help="Learning rate of optimizer")
parser.add_argument("--train_reconstructor", type=bool, default=False, help="Forced retraining the reconstruction model")
parser.add_argument("--train_classifier", type=bool, default=False, help="Forced retraining the classification model")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Dataset
'''
directory = '/data0/paired_dataset/'
paired_dataset = PairedDataset(directory, args.sample, args.he, cell_types_cz)
adata = paired_dataset.cell_select()
paired_dataset.draw_umaps_expression()
palette_he = paired_dataset.palette_he
train_loader, test_loader = paired_dataset.loaders(seed, args.batch_size)

'''
Reconstruction
'''
reconstruction = Reconstruction(seed, adata, args.sample, args.he)
reconstruction.load(train_loader, args.epochs, args.lr, train=args.train_reconstructor)
reconstruction.evaluate(test_loader)
#reconstruction.draw_umaps_embedding(palette_he)
reconstruction.draw_heatmap(cell_types_cz, palette_he)

'''
Classification
'''
for cell_type in ['cell_type_common', 'cell_subtype_st']:
    classification = Classification(adata, args.sample, args.he, cell_types_cz, cell_type)
    classification.load(train_loader, args.epochs, args.lr, train=args.train_classifier)
    classification.evaluate(test_loader)
