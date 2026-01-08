# SHEST
Single-cell-level artificial intelligence from haematoxylin and eosin morphology for cell type prediction and spatial transcriptomics reconstruction

## Prerequisites

|Package|Version|
|---|---|
|python|3.10|
|scanpy|1.10.4|
|spatialdata|0.5.0|
|torch|2.6.0+cu124|
|timm|1.0.15|

## Execution
* Command to infer from a WSI file
```
python he.py --wsi {svs file path}
```
At this point, Cellpose is used for nuclear segmentation. The output files are `h5ad` files with cell-level expression reconstruction and type prediction, and `png` files with nuclear boundaries colored.

## Advanced usage
### Command for model training
```
python main.py --mode train
```
The cell type annotation file `he_annotation.csv` must be located under the `DIRECTORY/dataset/PLATFORM/SOURCE/SAMPLE/annotation/` directory, with the schema `cell_id,group`.

### Command for model test
```
python main.py
```
