# SHEST
Single-cell-level artificial intelligence from haematoxylin and eosin morphology for cell type prediction and spatial transcriptomics reconstruction

## Prerequisites

|Package|Version|
|---|---|
|scanpy|1.10.4|
|spatialdata|0.3.0|
|torch|2.6.0+cu124|

## Execution

```
usage: main.py [-h] [--raw_directory RAW_DIRECTORY] [--directory DIRECTORY] [--platform PLATFORM]
               [--sources SOURCES [SOURCES ...]] [--samples SAMPLES [SAMPLES ...]] [--organ ORGAN]
               [--cell_type CELL_TYPE] [--sc_annotate] [--save_image] [--batch_size BATCH_SIZE]
               [--split SPLIT] [--epochs EPOCHS] [--lr LR] [--mode MODE]

Sample information and hyperparameters

options:
  -h, --help            show this help message and exit
  --raw_directory RAW_DIRECTORY
                        Directory of raw data (default: /data0/)
  --directory DIRECTORY
                        Working directory (default: /data0/crp/)
  --platform PLATFORM   Platform of spatial transcriptomics (default: Xenium_Prime)
  --sources SOURCES [SOURCES ...]
                        Data sources (default: ['10X'])
  --samples SAMPLES [SAMPLES ...]
                        Sample names (default: ['Human_Lung_Cancer'])
  --organ ORGAN         Organ of the sample (default: lung)
  --cell_type CELL_TYPE
                        Cell type to consider (default: cell_type)
  --sc_annotate         Force annotation on the cells with a single cell reference (default: False)
  --save_image          Force saving images (default: False)
  --batch_size BATCH_SIZE
                        Batch size of data loader (default: 512)
  --split SPLIT         spit ratio for training dataset (default: 0.8)
  --epochs EPOCHS       Number of epochs in training (default: 40)
  --lr LR               Learning rate of optimiser (default: 0.01)
  --mode MODE           train, test or infer (default: test)
```

### Command for training
```
python main.py --mode train
```
The `he_annotation.csv` file must be located in the `DIRECTORY/dataset/PLATFORM/SOURCE/SAMPLE/annotation/` directory, with the following schema:
```
cell_id,group
```

### Command for test
```
python main.py --platform Xenium_V1 --sample Human_Lung_Cancer_Addon
```
