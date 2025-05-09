# SHEST

## Prerequisites

|Package|Version|
|---|---|
|scanpy|1.10.4|
|spatialdata|0.3.0|
|torch|2.6.0+cu124|

## Execution

```
usage: main.py [-h] [--raw_directory RAW_DIRECTORY] [--directory DIRECTORY] [--platform PLATFORM] [--source SOURCE]
               [--sample SAMPLE] [--upper UPPER] [--filter] [--cell_type CELL_TYPE] [--sc_annotate]
               [--batch_size BATCH_SIZE] [--rotate] [--epochs EPOCHS] [--lr LR] [--train]

Sample information and hyperparameters

options:
  -h, --help            show this help message and exit
  --raw_directory RAW_DIRECTORY
                        Directory of raw data (default: /data0/)
  --directory DIRECTORY
                        Working directory (default: /data0/crp/)
  --platform PLATFORM   Platform of spatial transcriptomics (default: Xenium_Prime)
  --source SOURCE       Data source (default: 10X)
  --sample SAMPLE       Sample name (default: Human_Lung_Cancer)
  --upper UPPER         Upper limit of the one side length of the bounding rectangle of a cell within an H&E image (in
                        micrometers) (default: 18)
  --filter              Force filtration by the cell area (default: False)
  --cell_type CELL_TYPE
                        Cell type to consider (default: Cell_type)
  --sc_annotate         Force annotation on the cells with a single cell reference (default: False)
  --batch_size BATCH_SIZE
                        Batch size of data loader (default: 128)
  --rotate              Rotate the images in 0, 90, 180 and 270 degree (default: False)
  --epochs EPOCHS       Number of epochs in training (default: 30)
  --lr LR               Learning rate of optimizer (default: 0.01)
  --train               Force training the model (default: False)
```
