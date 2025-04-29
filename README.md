# SHEST

## Prerequisites

|Package|Version|
|---|---|
|scanpy|1.10.4|
|spatialdata|0.3.0|
|torch|2.6.0+cu124|

## Execution

### Preprocessing

```
usage: preprocess.py [-h] [--directory DIRECTORY] [--platform PLATFORM] [--sample SAMPLE]
                     [--cell_type CELL_TYPE] [--force_annotate] [--force_categorize]

Sample information

options:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        Directory of dataset (default: /data0/crp/dataset/)
  --platform PLATFORM   Platform of spatial transcriptomics (default: Xenium_Prime)
  --sample SAMPLE       Sample name (default: Human_Lung_Cancer)
  --cell_type CELL_TYPE
                        Cell type to consider (default: Cell_type)
  --force_annotate      If set, annotate the cells again (default: False)
  --force_categorize    If set, categorize the cells again (default: False)
```

### Modeling

```
usage: main.py [-h] [--directory DIRECTORY] [--platform PLATFORM] [--sample SAMPLE] [--he HE]
               [--cell_type CELL_TYPE] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
               [--lr_reconstructor LR_RECONSTRUCTOR] [--lr_classifier LR_CLASSIFIER]
               [--train_reconstructor] [--train_classifier]

Sample information and hyperparameters

options:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        Directory of dataset (default: /data0/crp/dataset/)
  --platform PLATFORM   Platform of spatial transcriptomics (default: Xenium_Prime)
  --sample SAMPLE       Sample name (default: Human_Lung_Cancer)
  --he HE               H&E images with side length (default: he70)
  --cell_type CELL_TYPE
                        A kind of cell typing to consider (default: Cell_type)
  --batch_size BATCH_SIZE
                        Batch size of data loader (default: 128)
  --epochs EPOCHS       Number of epochs in training (default: 20)
  --lr_reconstructor LR_RECONSTRUCTOR
                        Learning rate of optimizer for reconstructor (default: 0.01)
  --lr_classifier LR_CLASSIFIER
                        Learning rate of optimizer for classifier (default: 0.1)
  --train_reconstructor
                        If set, retrain the reconstruction model (default: False)
  --train_classifier    If set, retrain the classification model (default: False)
```
