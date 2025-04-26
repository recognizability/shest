import os
import random
import argparse
from tqdm import tqdm

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from config import seed, set_seed
from preprocess import cell_types_lung
from utils import PairedDataset
from model import Reconstruction, Classification

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample information and hyperparameters",                             
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--directory", type=str, default="/data0/crp/dataset/", help="Directory of dataset")
    parser.add_argument("--platform", type=str, default="Xenium_Prime", help="Platform of spatial transcriptomics")
    parser.add_argument("--sample", type=str, default="Human_Lung_Cancer", help="Sample name")
    parser.add_argument("--he", type=str, default="he70", help="H&E images with side length")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size of data loader")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs in training")
    parser.add_argument("--lr", type=int, default=0.01, help="Learning rate of optimizer")
    parser.add_argument("--train_reconstructor", action="store_true", help="If set, retrain the reconstruction model")
    parser.add_argument("--train_classifier", action="store_true", help="If set, retrain the classification model")
    args = parser.parse_args()

    if 'lung' in args.sample or 'Lung' in args.sample:
        cell_types = cell_types_lung

    paired_dataset = PairedDataset(args.directory, args.platform, args.sample, args.he, cell_types)
    adata = paired_dataset.cell_select()
    paired_dataset.draw_umaps_expression()
    palette_type = paired_dataset.palette_type
    train_loader, test_loader = paired_dataset.loaders(args.batch_size)

    reconstruction = Reconstruction(adata, args.platform, args.sample, args.he)
    reconstruction.load(train_loader, args.epochs, args.lr, train=args.train_reconstructor)
    reconstruction.evaluate(test_loader)
#    reconstruction.draw_umaps_embedding(palette_type)
    reconstruction.draw_heatmap(cell_types, palette_type)

    cell_type = 'Cell_type'
    classification = Classification(adata, args.platform, args.sample, args.he, cell_types, cell_type)
    classification.load(train_loader, args.epochs, args.lr, train=args.train_classifier)
    classification.evaluate(test_loader)
