# SHEST

## Prerequisites

|Package|Version|
|---|---|
|scanpy|1.10.4|
|spatialdata|0.3.0|
|torch|2.6.0+cu124|

## Execution

```
usage: main.py [-h] [--directory DIRECTORY] [--platform PLATFORM] [--source SOURCE] [--sample SAMPLE] [--he HE]
               [--cell_type CELL_TYPE] [--force_annotate] [--force_categorize] [--batch_size BATCH_SIZE]
               [--epochs EPOCHS] [--lr LR] [--train] [--rotate]

Sample information and hyperparameters

options:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        Directory of dataset (default: /data0/crp/dataset/)
  --platform PLATFORM   Platform of spatial transcriptomics (default: Xenium_Prime)
  --source SOURCE       Data source (default: 10X)
  --sample SAMPLE       Sample name (default: Human_Lung_Cancer)
  --he HE               H&E images with side length (default: he84)
  --cell_type CELL_TYPE
                        Cell type to consider (default: Cell_type)
  --force_annotate      If set, annotate the cells again (default: False)
  --force_categorize    If set, categorize the cells again (default: False)
  --batch_size BATCH_SIZE
                        Batch size of data loader (default: 128)
  --epochs EPOCHS       Number of epochs in training (default: 20)
  --lr LR               Learning rate of optimizer (default: 0.01)
  --train               Retrain the model (default: False)
  --rotate              Rotate the images in 0, 90, 180 and 270 degree (default: False)
```
