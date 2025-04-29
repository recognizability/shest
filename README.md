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
usage: preprocess.py [-h] [--directory DIRECTORY] [--platform PLATFORM] [--sample SAMPLE] [--force_annotate]
                     [--force_categorize]

Sample information

options:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        Directory of dataset (default: /data0/crp/dataset/)
  --platform PLATFORM   Platform of spatial transcriptomics (default: Xenium_Prime)
  --sample SAMPLE       Sample name (default: Human_Lung_Cancer)
  --force_annotate      If set, annotate again (default: False)
  --force_categorize    If set, categorize again (default: False)
```

### Modeling

```
usage: main.py [-h] [--directory DIRECTORY] [--platform PLATFORM] [--sample SAMPLE] [--he HE]
               [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--train_reconstructor] [--train_classifier]

Sample information and hyperparameters

options:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        Directory of dataset (default: /data0/crp/dataset/)
  --platform PLATFORM   Platform of spatial transcriptomics (default: Xenium_Prime)
  --sample SAMPLE       Sample name (default: Human_Lung_Cancer)
  --he HE               H&E images with side length (default: he70)
  --batch_size BATCH_SIZE
                        Batch size of data loader (default: 128)
  --epochs EPOCHS       Number of epochs in training (default: 20)
  --lr LR               Learning rate of optimizer (default: 0.01)
  --train_reconstructor
                        If set, retrain the reconstruction model (default: False)
  --train_classifier    If set, retrain the classification model (default: False)
```
