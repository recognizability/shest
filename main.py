import argparse

from config import seed, set_seed, Config
from preprocess import Preprocessing
from model import Modeling

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample information and hyperparameters",                             
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--raw_directory", type=str, default="/data0/", help="Directory of raw data")
    parser.add_argument("--directory", type=str, default="/data0/crp/", help="Working directory ")
    parser.add_argument("--platform", type=str, default="Xenium_Prime", help="Platform of spatial transcriptomics")
    parser.add_argument("--source", type=str, default="10X", help="Data source")
    parser.add_argument("--sample", type=str, default="Human_Lung_Cancer", help="Sample name")
    parser.add_argument("--upper", type=int, default=18, help="Upper limit of the one side length of the bounding rectangle of a cell within an H&E image (in micrometers)")
    parser.add_argument("--filter", action="store_true", help="Force filtration by the cell area")
    parser.add_argument("--variable", action="store_true", help="Variable image crop size")
    parser.add_argument("--cell_type", type=str, default="Cell_type", help="Cell type to consider")
    parser.add_argument("--sc_annotate", action="store_true", help="Force annotation on the cells with a single cell reference")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size of data loader")
    parser.add_argument("--rotate", action="store_true", help="Rotate the images in 0, 90, 180 and 270 degree")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs in training")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate of optimizer")
    parser.add_argument("--train", action="store_true", help="Force training the model")
    args = parser.parse_args()
    print(args)

    config = Config(args)

    Preprocessing(args, config)

    modeling = Modeling(args, config)
    modeling.evaluate()
    modeling.draw_heatmap()
    modeling.draw_confusion_matrix()
    modeling.draw_umaps_embedding()
