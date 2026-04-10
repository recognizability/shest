# SHEST
Single-cell-level artificial intelligence from haematoxylin and eosin morphology for cell type prediction and spatial transcriptomics reconstruction

## Installation
```
conda create -n shest python=3.10 scanpy==1.10.4 spatialdata==0.5.0 tacco libpysal esda -c conda-forge -c bioconda
conda activate shest
```
```
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install spatialdata-io timm==1.0.15 opencv-python squidpy cellpose openslide-python openslide-bin
```
```
git clone https://github.com/recognizability/shest.git
cd shest
```

## Execution
Command to infer from a WSI file
```
python he.py --wsi {svs file path} --directory {path}
```
* The `file path` of the WSI file and the `path` of the processing directory must be specified as command-line arguments. Model weight `*.pth` files must be located under `models` within the `path`, and result files are generated in the `he` directory under the `path`.
* During processing, Cellpose is used for nuclear segmentation. The output files consist of `h5ad` files with cell-level expression reconstruction and type prediction, and a `geojson` file containing cell types and their color information. The colors by cell type are as follows:

|Cell type|Color|
|---|---|
|`Alveolar_cell`|Pink|
|`Tumor_cell_LUAD`|Red|
|`Macrophage`|Orange|
|`Endothelial_cell`|Yellow|
|`Fibroblast`|Green|
|`Lymphocyte`|Blue|

## Advanced usage
### Command for model test
```
python main.py
```

### Command for model training
```
python main.py --mode train
```
The cell type annotation file `he_annotation.csv` must be located under the `DIRECTORY/dataset/PLATFORM/SOURCE/SAMPLE/annotation/` directory, with the schema `cell_id,group`.

## Citation
<https://academic.oup.com/bib/article/27/1/bbag037/8488669>
```
@article{jeong2026shest,
  title={SHEST: single-cell-level artificial intelligence from haematoxylin and eosin morphology for cell-type prediction and spatial transcriptomics reconstruction},
  author={Jeong, Hoyeon and Oh, Junghan and Lee, Donggeon and Kang, Jae Hwan and Choi, Yoon-La},
  journal={Briefings in Bioinformatics},
  volume={27},
  number={1},
  pages={bbag037},
  year={2026},
  publisher={Oxford University Press},
  keywords={published, selected},
}
```
