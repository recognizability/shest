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
    parser.add_argument("--cell_type", type=str, default="Cell_type", help="A kind of cell typing to consider")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size of data loader")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs in training")
    parser.add_argument("--lr_reconstructor", type=float, default=0.01, help="Learning rate of optimizer for reconstructor")
    parser.add_argument("--lr_classifier", type=float, default=0.01, help="Learning rate of optimizer for classifier")
    parser.add_argument("--train_reconstructor", action="store_true", help="Retrain the reconstruction model")
    parser.add_argument("--train_classifier", action="store_true", help="Retrain the classification model")
    parser.add_argument("--rotate", action="store_true", help="Rotate the images in 0, 90, 180 and 270 degree")
    args = parser.parse_args()

    if 'lung' in args.sample or 'Lung' in args.sample:
        cell_types = cell_types_lung

    if not args.rotate:
        angles = [0]
    else:
        angles = [0, 90, 180, 270]

    paired_dataset = PairedDataset(args.directory, args.platform, args.sample, args.he, args.cell_type, cell_types, angles)
#    paired_dataset.draw_umaps_expression()
    palette_type = paired_dataset.palette_type
    var_names, classes, train_loader, test_loader = paired_dataset.get_dataloaders(args.batch_size)

    reconstruction = Reconstruction(args.platform, args.sample, args.he, args.cell_type, angles, var_names, classes)
    reconstruction.load(train_loader, args.epochs, args.lr_reconstructor, train=args.train_reconstructor)
    reconstruction.evaluate(test_loader)
#    reconstruction.draw_umaps_embedding(palette_type)
    reconstruction.draw_heatmap(cell_types, palette_type)

    classification = Classification(args.platform, args.sample, args.he, args.cell_type, cell_types, angles, var_names, classes)
    classification.load(train_loader, args.epochs, args.lr_classifier, train=args.train_classifier)
    classification.evaluate(test_loader)
