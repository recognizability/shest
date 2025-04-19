# SHEST

## Prerequisites

|Package|Version|
|---|---|
|scanpy|1.10.4|
|spatialdata|0.3.0|
|torch|2.6.0+cu124|

## Execution

```
python main.py
```

```
Sample information and hyperparameters

options:
  -h, --help            show this help message and exit
  --sample SAMPLE       Sample name consisting of platform, panel size, and organ (default: 10x_5k_lung)
  --he HE               Side length of H&E image (default: he200)
  --batch_size BATCH_SIZE
                        Batch size of data loader (default: 512)
  --epochs EPOCHS       Number of epochs in training (default: 20)
  --lr LR               Learning rate of optimizer (default: 0.01)
  --force FORCE         Forced training (default: False)
```
