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
