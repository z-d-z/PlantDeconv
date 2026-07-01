# PlantDeconv

PlantDeconv is a structure-guided probabilistic deconvolution framework for analyzing cell-type composition in plant spatial transcriptomics data.

## Overview

Spatial transcriptomics provides a powerful strategy for resolving plant cell composition and developmental states within native tissue contexts. However, many sequencing-based spatial platforms measure multiple cells within each spot, making cell-type deconvolution necessary. Existing deconvolution methods are mainly designed for animal tissues or general tissue assumptions, and often do not explicitly use plant-specific spatial structures such as anatomical regions, layered organization, cell-wall-constrained spatial patterns, tissue boundaries, and continuous developmental gradients.

PlantDeconv addresses this problem by integrating single-cell reference data, spatial transcriptomics data, region-aware priors, expression-aware spatial continuity, differentiation-gradient constraints, and adaptive spot-level regularization into a unified probabilistic deconvolution model.

## Model Framework

![PlantDeconv framework](framework.png)

PlantDeconv takes two main inputs:

- A single-cell RNA-seq reference dataset with cell-type annotations
- A spatial transcriptomics dataset with spot-level gene expression and spatial coordinates

The framework first selects marker genes from the single-cell reference and identifies shared training genes between the single-cell and spatial datasets. It then builds spatial-region metadata, estimates cell-type-to-region priors using marker-driven pseudobulk similarity, and performs cluster-to-spot probabilistic mapping.

The core model estimates a cell-type composition matrix for each spatial spot and reconstructs spatial gene expression using the learned mapping. PlantDeconv introduces three structure-aware regularization components:

1. **Expression-aware spatial continuity**  
   Encourages neighboring spots with similar expression profiles to have similar cell-type compositions.

2. **Differentiation-gradient constraint**  
   Models continuous developmental transitions along spatial or anatomical gradients.

3. **Adaptive spot regularization**  
   Applies stronger regularization in homogeneous tissue regions and weaker regularization near complex boundaries.

These components help reduce unrealistic spatial diffusion and improve anatomical consistency in plant tissues.

## Benchmark and Testing Strategy

![Pseudo-spot benchmark](benchmark.png)

The model can be evaluated using pseudo-spot benchmarks generated from high-resolution spatial annotations. In this testing strategy, annotated spatial bins are merged into larger pseudo spots, such as 32 μm or 64 μm pseudo spots. Since the original cell-type annotations are known, the true cell-type composition of each pseudo spot can be calculated and used as ground truth.

PlantDeconv predicts the cell-type proportions for each pseudo spot, and the predicted composition is compared with the known ground-truth composition. This enables quantitative evaluation of deconvolution performance under different spatial resolutions and spot sizes.

## Installation

We recommend creating a clean conda environment before running PlantDeconv.

```bash
git clone https://github.com/z-d-z/PlantDeconv.git
cd PlantDeconv

conda create -n plantdeconv python=3.10 -y
conda activate plantdeconv

pip install -r requirements.txt
```

If GPU acceleration is required, please make sure that the installed PyTorch version matches your CUDA version. For example, for CUDA 12.1:

```bash
pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

For CPU-only usage, install the CPU-compatible PyTorch version instead.

---

## Quick Start

The main pipeline is implemented in `main.py`.

Before running PlantDeconv, edit the configuration section at the top of `main.py` and set the paths to your own data:

```python
SC_H5AD = Path("path/to/single_cell_reference.h5ad")
ST_H5AD = Path("path/to/spatial_data.h5ad")
HE_IMAGE_PATH = Path("path/to/tissue_hires_image.png")
OUTPUT_DIR = Path("path/to/output_directory")
```

Then run:

```bash
python main.py
```

PlantDeconv automatically uses GPU when `torch.cuda.is_available()` is true; otherwise, it falls back to CPU.

---

## Input Requirements

**Single-cell AnnData (`adata_sc`)**

- Expression matrix in `.X` or `.layers['counts']`
- Cell-type annotation in `obs[cluster_label]`, default: `cell_type`
- Precomputed marker-gene ranking in `uns[marker_key]`, default: `wilcoxon`

If marker genes have not been precomputed, they can be generated using Scanpy:

```python
import scanpy as sc

sc.tl.rank_genes_groups(
    adata_sc,
    groupby="cell_type",
    method="wilcoxon"
)

adata_sc.write_h5ad("single_cell_reference_with_markers.h5ad")
```

**Spatial AnnData (`adata_sp`)**

- Expression matrix in `.X` or `.layers['counts']`
- Spatial coordinates in `obsm['spatial']`
- Optional region labels, such as `domain` or `cluster`
- Optional H&E image and `scalefactors_json.json` for spatial visualization

---

## Main Outputs

All results are saved to `outputs_plantdeconv/` by default.

Main output files include:

- `plantdeconv_spatial_annotated.h5ad`: spatial AnnData with predicted cell-type proportions stored in `obsm['plantdeconv_ct_pred']`
- `plantdeconv_celltype_proportions.csv`: predicted cell-type proportions for each spot
- `plantdeconv_training_history.csv`: optimization loss history
- `plantdeconv_spot_summary.csv`: top predicted cell types and spot-level summary statistics
- `plantdeconv_ct_layer_prior.csv`: inferred cell-type-to-region prior
- `plantdeconv_top_celltype_spatial.png`: spatial visualization of dominant predicted cell types
- `plantdeconv_pie_map.png`: pie-chart map of predicted cell-type composition
- `plantdeconv_summary.json`: hyperparameters, data shapes, and summary metrics

---

## Notes

PlantDeconv expects the single-cell and spatial datasets to share a sufficient number of genes. The value of `cluster_label` must match the cell-type annotation column in `adata_sc.obs`, and the value of `marker_key` must match the key used by `sc.tl.rank_genes_groups`.

Region-based regularization is optional and is most useful when reliable spatial region annotations are available.
## Citation

If you use PlantDeconv, please cite this repository and the associated manuscript. Citation information will be updated after publication.
