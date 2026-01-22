import argparse
import pprint
import torch

from config import Config
from data import Dataset
from model import Modeling

def set_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sample information and hyperparameters",                             
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--raw_directory", type=str, default="/data0/", help="Directory of raw data")
    parser.add_argument("--directory", type=str, default="/data0/crp/", help="Working directory ")
    parser.add_argument("--platform", type=str, default="Xenium_Prime", help="Platform of spatial transcriptomics")
    parser.add_argument("--sources", type=str, nargs="+", default=["10X", "SMC", "SMC"], help="Data sources")
    parser.add_argument("--samples", type=str, nargs="+", default=["Human_Lung_Cancer", "03320", "03331"], help="Sample names")
    parser.add_argument("--organ", type=str, default="lung", help="Organ of the sample")
    parser.add_argument("--cell_type", type=str, default="cell_type", help="Cell type to consider")
    parser.add_argument("--sc_annotate", action="store_true", help="Force annotation on the cells with a single cell reference")
    parser.add_argument("--save_images", action="store_true", help="Force saving images")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size of data loader")
    parser.add_argument("--split", type=float, default=0.8, help="Spit ratio for training dataset")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs in training")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate of optimiser")
    parser.add_argument("--mode", type=str, default="test", help="Choose between train, test or infer")
    args, remaining = parser.parse_known_args(argv)
    return args, remaining

if __name__ == "__main__":
    args, _ = set_args()
    config = Config(args)
    dataset = Dataset(args, config)
    modeling = Modeling(args, config, dataset)
    modeling.draw()
