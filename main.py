import argparse
import pprint
import torch

from config import seed, set_seed, Config
from dataset import Dataset, MergedDataset
from model import Modeling

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample information and hyperparameters",                             
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--raw_directory", type=str, default="/data0/", help="Directory of raw data")
    parser.add_argument("--directory", type=str, default="/data0/crp/", help="Working directory ")
    parser.add_argument("--platform", type=str, default="Xenium_Prime", help="Platform of spatial transcriptomics")
    parser.add_argument("--sources", type=str, nargs="+", default=["10X"], help="Data sources")
    parser.add_argument("--samples", type=str, nargs="+", default=["Human_Lung_Cancer"], help="Sample names")
    parser.add_argument("--organ", type=str, default="lung", help="Organ of the sample")
    parser.add_argument("--filter", action="store_true", help="Force filtration by the cell area")
    parser.add_argument("--original_image", action="store_true", help="Use the original imagem not the quadruple tiles")
    parser.add_argument("--cell_type", type=str, default="cell_type", help="Cell type to consider")
    parser.add_argument("--sc_annotate", action="store_true", help="Force annotation on the cells with a single cell reference")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size of data loader")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs in training")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate of optimizer")
    parser.add_argument("--mode", type=str, default="test", help="train, test or infer")
    args = parser.parse_args()
    print(args)

    config = Config(args)
    merged_dataset = MergedDataset(args, config)

    modeling = Modeling(args, config, merged_dataset)
    modeling.evaluate()
    modeling.draw_confusion_matrix()
    if not 'subtype' in args.cell_type:
        modeling.draw_heatmap()
#        modeling.draw_umaps_embedding()
