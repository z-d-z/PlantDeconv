#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlantDeconv v2.1 — Differentiation-continuity-aware spatial deconvolution
for plant tissues.

Complete pipeline:
  1. Load & normalise single-cell and spatial transcriptomics data.
  2. Select marker genes from differential expression results.
  3. Run PlantDeconv preprocessing (shared gene detection, density priors).
  4. Build spot-level spatial-region metadata (automatic ordering).
  5. Infer cell-type-to-region priors via marker-driven pseudobulk similarity.
  6. Run differentiation-continuity-aware deconvolution mapping.
  7. Project cell-type annotations & gene expression onto space.
  8. Evaluate results (gene scores, region consistency, spot summary).
  9. Save all outputs.

v2.1 removes explicit topology selection and instead uses three layers of
differentiation-continuity regularisation that work universally for any
plant tissue type.

Usage:
    python run_plantdeconv.py

    Modify the paths and hyperparameters below to match your data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse

# ---- Import PlantDeconv ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import PlantDeconv as pdv


# ============================================================================
# Configuration
# ============================================================================

# ---- Input data paths ----
# SC_H5AD = Path(
#     r"../../sc_data\IN4\PGT\PGT_clustered.h5ad"
# )
# ST_H5AD = Path(
#     r"../../st_data\poplar_4\poplar_4_with_clusters.h5ad"
# )
# HE_IMAGE_PATH = Path(
#     r"../../st_data\poplar_4\spatial\tissue_hires_image.png"
# )
SC_H5AD = Path(
    r"D:\Anaconda\anaconda\envs\PlantST\PlantST-main\Deconv\sc_data\soybean\GSE270392_Gm_atlas_Early_maturation_stage_seeds.with_real_celltype.h5ad"
)
ST_H5AD = Path(
    r"D:\Anaconda\anaconda\envs\PlantST\PlantST-main\Deconv\st_data\soybean\大豆_2024\slice1.h5ad"
)
HE_IMAGE_PATH = Path(
    r"D:\Anaconda\anaconda\envs\PlantST\PlantST-main\Deconv\st_data\soybean\大豆_2024\GSM8341875_ES_A_tissue_hires_image.png"
)
# ---- Output directory ----
OUTPUT_DIR = SCRIPT_DIR / "大豆_2024"

# ---- Biological annotation ----
CLUSTER_LABEL = "celltype"
LAYER_LABEL_CANDIDATES = ("cell_types", "domain", "cluster", "celltype", "seurat_clusters")
MARKER_KEY = "wilcoxon"
N_MARKERS_PER_CELLTYPE = 80
MARKER_MIN_LOGFC = 1.0
MARKER_MAX_PVAL_ADJ = 0.05

# ---- Normalisation ----
NORMALIZE_INPUTS = True
TARGET_SUM = 1e4

# ---- Training hyperparameters ----
SCALE_CLUSTERS = False
DENSITY_PRIOR = "uniform"
RANDOM_STATE = 42
NUM_EPOCHS = 1800
LEARNING_RATE = 0.08

# ---- Loss term weights ----
LAMBDA_R = 1e-3                # Entropy regulariser
LAMBDA_SPOT_ENTROPY = 0.20     # Spot cell-type sparsity
LAMBDA_CT_ISLANDS = 0.05       # Cell-type island enforcement
LAMBDA_NEIGHBORHOOD_G1 = 0.20  # Neighbourhood gene expression similarity

# Region-based penalties (optional — work best when region annotations exist)
LAMBDA_LAYER_PRIOR = 0.3      # Region prior KL divergence
LAMBDA_OUT_OF_BAND = 0.2      # Out-of-band penalty
LAMBDA_LAYER_DISTANCE = 0.1   # Region distance penalty

# Differentiation-continuity penalties (core v2.1 — universal)
# Three-layer framework capturing plant cell differentiation continuity:
#   Layer 1: Expression-aware spatial continuity
#   Layer 2: Anisotropic differentiation gradient
#   Layer 3: Adaptive per-spot regularisation (automatic)
LAMBDA_SPATIAL_CONTINUITY = 0.30       # Layer 1 weight
LAMBDA_DIFFERENTIATION_GRADIENT = 0.20 # Layer 2 weight
CONTINUITY_NEIGHBORS = 6               # KNN for all three layers
EXPRESSION_WEIGHT = 0.5                # Layer 1: spatial vs expression balance
GRADIENT_SIGMA = 2.0                   # Layer 2: differentiation kernel bandwidth
ADAPTIVE_FLOOR = 0.1                   # Layer 3: min regularisation at boundaries
ADAPTIVE_CEILING = 1.0                 # Layer 3: max regularisation in homogeneous areas

# ---- Layer prior construction ----
PRIOR_TEMPERATURE = 0.8
LAYER_BANDWIDTH = 3
SECTION_THRESHOLD_FACTOR = 2.5
SECTION_NEIGHBORS = 6
MIN_MARKER_GENES_PER_CT = 12

# ---- Plotting ----
PLOT_TOP_N_CELL_TYPES = 6
PLOT_IMG_KEY_PREFERENCE = ("hires", "lowres")


# ============================================================================
# Helper functions
# ============================================================================

def log(message: str) -> None:
    print(f"[PlantDeconv] {message}", flush=True)


def assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def sanitize_name(name: str) -> str:
    safe = str(name).replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe = safe.replace("(", "").replace(")", "").replace(":", "_")
    return safe


# ---- H5AD reading helpers ----

def _decode_h5_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_h5_scalar(value.item())
    return value


def _decode_h5_array(values):
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "O"}:
        flat = [_decode_h5_scalar(x) for x in arr.reshape(-1).tolist()]
        return np.asarray(flat, dtype=object).reshape(arr.shape)
    return arr


def _read_h5ad_csr(group: h5py.Group):
    shape = tuple(int(x) for x in group.attrs["shape"])
    data = np.asarray(group["data"][...])
    indices = np.asarray(group["indices"][...], dtype=np.int32)
    indptr = np.asarray(group["indptr"][...], dtype=np.int32)
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def _read_h5ad_dataframe(group: h5py.Group) -> pd.DataFrame:
    index_key = _decode_h5_scalar(group.attrs["_index"])
    index = _decode_h5_array(group[index_key][...]).tolist()
    column_order = [_decode_h5_scalar(x) for x in group.attrs["column-order"]]

    data = {}
    for col in column_order:
        item = group[str(col)]
        if isinstance(item, h5py.Group) and item.attrs.get("encoding-type") == "categorical":
            codes = np.asarray(item["codes"][...], dtype=np.int64)
            categories = _decode_h5_array(item["categories"][...]).tolist()
            ordered = bool(item.attrs.get("ordered", False))
            data[str(col)] = pd.Categorical.from_codes(
                codes, categories=categories, ordered=ordered,
            )
        else:
            values = _decode_h5_array(item[...])
            if isinstance(values, np.ndarray) and values.ndim == 0:
                values = np.repeat(_decode_h5_scalar(values), len(index))
            data[str(col)] = values

    return pd.DataFrame(data, index=pd.Index(index, name=str(index_key)))


# ============================================================================
# Data loading
# ============================================================================

def load_sc(path: Path) -> sc.AnnData:
    log(f"Loading single-cell data: {path}")
    backed = sc.read_h5ad(path, backed="r")
    try:
        if "counts" in backed.layers:
            x = backed.layers["counts"]
        else:
            x = backed.X.to_memory() if hasattr(backed.X, "to_memory") else backed.X.copy()
        obs = backed.obs.copy()
        var = backed.var.copy()
        uns = {}
        if MARKER_KEY in backed.uns:
            uns[MARKER_KEY] = backed.uns[MARKER_KEY].copy()
    finally:
        if getattr(backed, "file", None) is not None:
            backed.file.close()

    adata = sc.AnnData(X=x, obs=obs, var=var, uns=uns)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata


def load_st(path: Path) -> sc.AnnData:
    log(f"Loading spatial data: {path}")
    with h5py.File(path, "r") as f:
        if "layers" in f and "counts" in f["layers"]:
            x = _read_h5ad_csr(f["layers"]["counts"])
        else:
            x = _read_h5ad_csr(f["X"])

        obs = _read_h5ad_dataframe(f["obs"])
        var = _read_h5ad_dataframe(f["var"])

        obsm = {}
        if "obsm" in f and "spatial" in f["obsm"]:
            obsm["spatial"] = np.asarray(f["obsm"]["spatial"][...])

    adata = sc.AnnData(X=x, obs=obs, var=var, obsm=obsm, uns={})
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata


# ============================================================================
# Preprocessing helpers
# ============================================================================

def normalize_log1p(adata: sc.AnnData, label: str) -> sc.AnnData:
    adata = adata.copy()
    if NORMALIZE_INPUTS:
        log(f"{label}: normalize_total(target_sum={TARGET_SUM}) + log1p")
        sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
        sc.pp.log1p(adata)
    return adata


def select_marker_genes(adata_sc: sc.AnnData) -> Tuple[List[str], Dict[str, List[str]]]:
    if MARKER_KEY not in adata_sc.uns:
        raise KeyError(
            f"Missing adata_sc.uns['{MARKER_KEY}']; "
            f"run sc.tl.rank_genes_groups() first."
        )

    groups = sorted(
        adata_sc.obs[CLUSTER_LABEL].astype(str).unique(),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    marker_genes: List[str] = []
    markers_by_ct: Dict[str, List[str]] = {}
    per_group_counts: Dict[str, int] = {}

    for group in groups:
        df = sc.get.rank_genes_groups_df(adata_sc, group=group, key=MARKER_KEY)
        if "logfoldchanges" in df.columns:
            df = df[df["logfoldchanges"].fillna(0) >= MARKER_MIN_LOGFC]
        if "pvals_adj" in df.columns:
            df = df[df["pvals_adj"].fillna(1.0) <= MARKER_MAX_PVAL_ADJ]
        genes = (
            df["names"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(N_MARKERS_PER_CELLTYPE)
            .tolist()
        )
        markers_by_ct[str(group)] = genes
        per_group_counts[str(group)] = len(genes)
        marker_genes.extend(genes)

    marker_genes = list(dict.fromkeys(marker_genes))
    log(f"Selected {len(marker_genes)} unique marker genes for training.")
    log(f"Per-cell-type marker counts: {per_group_counts}")
    return marker_genes, markers_by_ct


def run_pp_adatas(adata_sc: sc.AnnData, adata_sp: sc.AnnData, genes: List[str]) -> None:
    try:
        pdv.pp_adatas(adata_sc, adata_sp, genes=genes, gene_to_lowercase=True)
    except ModuleNotFoundError as exc:
        if exc.name != "squidpy" or "spatial" not in adata_sp.obsm:
            raise
        log("squidpy unavailable; running preprocessing without precomputed spatial neighbours.")
        spatial_backup = adata_sp.obsm["spatial"].copy()
        del adata_sp.obsm["spatial"]
        pdv.pp_adatas(adata_sc, adata_sp, genes=genes, gene_to_lowercase=True)
        adata_sp.obsm["spatial"] = spatial_backup


# ============================================================================
# Visualisation helpers
# ============================================================================

def ensure_spatial_image(adata_sp: sc.AnnData, image_path: Path) -> Optional[str]:
    if "spatial" not in adata_sp.uns or not adata_sp.uns["spatial"]:
        return None
    library_id = list(adata_sp.uns["spatial"].keys())[0]
    block = adata_sp.uns["spatial"][library_id]
    block.setdefault("images", {})
    block.setdefault("scalefactors", {})

    if image_path.exists() and "hires" not in block["images"]:
        block["images"]["hires"] = mpimg.imread(image_path)

    scale_json = image_path.parent / "scalefactors_json.json"
    if scale_json.exists():
        with open(scale_json, "r", encoding="utf-8") as f:
            block["scalefactors"].update(json.load(f))

    for img_key in PLOT_IMG_KEY_PREFERENCE:
        if img_key in block["images"]:
            return img_key
    return None


def save_spatial_preview(adata_sp: sc.AnnData, output_dir: Path) -> None:
    if "plantdeconv_ct_pred" not in adata_sp.obsm or "spatial" not in adata_sp.obsm:
        log("Skipping spatial preview (missing predictions or coordinates).")
        return

    ct_pred = adata_sp.obsm["plantdeconv_ct_pred"].copy()
    if not isinstance(ct_pred, pd.DataFrame):
        ct_pred = pd.DataFrame(ct_pred, index=adata_sp.obs_names)
    top_cols = ct_pred.sum(axis=0).sort_values(ascending=False).head(PLOT_TOP_N_CELL_TYPES).index

    color_keys = []
    for ct in top_cols:
        key = f"pd_{sanitize_name(ct)}"
        adata_sp.obs[key] = ct_pred.loc[adata_sp.obs_names, ct].astype(float)
        color_keys.append(key)

    img_key = ensure_spatial_image(adata_sp, HE_IMAGE_PATH)
    if img_key is None:
        log("No H&E image available for spatial preview.")
        return

    sc.pl.spatial(
        adata_sp,
        color=color_keys,
        img_key=img_key,
        cmap="viridis",
        show=False,
    )
    out_path = output_dir / "plantdeconv_top_celltype_spatial.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"Spatial preview saved: {out_path}")


def training_history_to_frame(training_history: dict) -> pd.DataFrame:
    if not training_history:
        return pd.DataFrame()
    max_len = max(len(v) for v in training_history.values())
    data = {}
    for key, values in training_history.items():
        values = list(values)
        if len(values) < max_len:
            values = values + [np.nan] * (max_len - len(values))
        data[key] = values
    return pd.DataFrame(data)


# ============================================================================
# Main pipeline
# ============================================================================

def main() -> None:
    """Run the full PlantDeconv v2.1 pipeline."""

    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    assert_exists(SC_H5AD, "Single-cell h5ad")
    assert_exists(ST_H5AD, "Spatial h5ad")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sc.settings.figdir = str(OUTPUT_DIR)
    sc.settings.set_figure_params(dpi=150, facecolor="white", frameon=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"PlantDeconv v{pdv.__version__}")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Device: {device}")
    log("Regularisation mode: differentiation-continuity-aware (structure-agnostic)")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    adata_sc = load_sc(SC_H5AD)
    adata_sp = load_st(ST_H5AD)

    if CLUSTER_LABEL not in adata_sc.obs.columns:
        raise KeyError(f"Missing adata_sc.obs['{CLUSTER_LABEL}']")
    adata_sc.obs[CLUSTER_LABEL] = adata_sc.obs[CLUSTER_LABEL].astype(str)

    log(f"Single-cell: {adata_sc.n_obs} cells, {adata_sc.n_vars} genes")
    log(f"Spatial: {adata_sp.n_obs} spots, {adata_sp.n_vars} genes")
    log(f"Cell types: {adata_sc.obs[CLUSTER_LABEL].nunique()}")

    # ------------------------------------------------------------------
    # 2. Normalise
    # ------------------------------------------------------------------
    adata_sc = normalize_log1p(adata_sc, "sc")
    adata_sp = normalize_log1p(adata_sp, "st")

    # ------------------------------------------------------------------
    # 3. Select marker genes & preprocess
    # ------------------------------------------------------------------
    marker_genes, markers_by_ct = select_marker_genes(adata_sc)
    run_pp_adatas(adata_sc, adata_sp, marker_genes)
    log(f"Training genes after preprocessing: {len(adata_sc.uns['training_genes'])}")
    log(f"Overlap genes after preprocessing:  {len(adata_sc.uns['overlap_genes'])}")

    # ------------------------------------------------------------------
    # 4. Build spatial-region metadata (automatic ordering)
    # ------------------------------------------------------------------
    spot_meta, layer_order, layer_key = pdv.build_spot_layer_metadata(
        adata_sp,
        layer_key_candidates=LAYER_LABEL_CANDIDATES,
        section_threshold_factor=SECTION_THRESHOLD_FACTOR,
        section_neighbors=SECTION_NEIGHBORS,
    )
    ordering_strategy = spot_meta.attrs.get("ordering_strategy", "auto")
    log(f"Region annotation key: {layer_key}")
    log(f"Region ordering strategy (auto-detected): {ordering_strategy}")
    log(f"Region order: {layer_order}")

    # ------------------------------------------------------------------
    # 5. Build cell-type-to-region prior
    # ------------------------------------------------------------------
    prior_result = pdv.build_ct_layer_prior(
        adata_sc=adata_sc,
        adata_sp=adata_sp,
        cluster_label=CLUSTER_LABEL,
        genes=adata_sc.uns["training_genes"],
        spot_meta=spot_meta,
        layer_order=layer_order,
        markers_by_ct=markers_by_ct,
        prior_temperature=PRIOR_TEMPERATURE,
        layer_bandwidth=LAYER_BANDWIDTH,
        min_marker_genes=MIN_MARKER_GENES_PER_CT,
    )
    log("Cell-type-to-region prior matrix constructed.")

    # ------------------------------------------------------------------
    # 6. Differentiation-continuity-aware deconvolution mapping
    # ------------------------------------------------------------------
    log("Starting PlantDeconv differentiation-continuity-aware mapping...")
    log(f"  Layer 1 (spatial continuity): lambda={LAMBDA_SPATIAL_CONTINUITY}, "
        f"expression_weight={EXPRESSION_WEIGHT}")
    log(f"  Layer 2 (differentiation gradient): lambda={LAMBDA_DIFFERENTIATION_GRADIENT}, "
        f"sigma={GRADIENT_SIGMA}")
    log(f"  Layer 3 (adaptive regularisation): floor={ADAPTIVE_FLOOR}, "
        f"ceiling={ADAPTIVE_CEILING}")

    adata_map = pdv.map_clusters_to_space_layeraware(
        adata_sc=adata_sc,
        adata_sp=adata_sp,
        cluster_label=CLUSTER_LABEL,
        ct_layer_prior=prior_result.prior,
        spot_meta=spot_meta,
        device=device,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        scale=SCALE_CLUSTERS,
        density_prior=DENSITY_PRIOR,
        lambda_r=LAMBDA_R,
        lambda_spot_entropy=LAMBDA_SPOT_ENTROPY,
        lambda_ct_islands=LAMBDA_CT_ISLANDS,
        lambda_neighborhood_g1=LAMBDA_NEIGHBORHOOD_G1,
        # Region-based penalties (optional)
        lambda_layer_prior=LAMBDA_LAYER_PRIOR,
        lambda_out_of_band=LAMBDA_OUT_OF_BAND,
        lambda_layer_distance=LAMBDA_LAYER_DISTANCE,
        # Differentiation-continuity penalties (core v2.1)
        lambda_spatial_continuity=LAMBDA_SPATIAL_CONTINUITY,
        lambda_differentiation_gradient=LAMBDA_DIFFERENTIATION_GRADIENT,
        continuity_neighbors=CONTINUITY_NEIGHBORS,
        expression_weight=EXPRESSION_WEIGHT,
        gradient_sigma=GRADIENT_SIGMA,
        adaptive_floor=ADAPTIVE_FLOOR,
        adaptive_ceiling=ADAPTIVE_CEILING,
        random_state=RANDOM_STATE,
        verbose=True,
    )
    log("Mapping completed.")

    # ------------------------------------------------------------------
    # 7. Project annotations & gene expression
    # ------------------------------------------------------------------
    pdv.project_cell_annotations(
        adata_map=adata_map,
        adata_sp=adata_sp,
        annotation=CLUSTER_LABEL,
    )
    adata_ge = pdv.project_genes(
        adata_map=adata_map,
        adata_sc=adata_sc,
        cluster_label=CLUSTER_LABEL,
        scale=SCALE_CLUSTERS,
    )
    df_compare = pdv.compare_spatial_geneexp(
        adata_ge=adata_ge,
        adata_sp=adata_sp,
        adata_sc=adata_sc,
    )
    log(f"Mean training gene score: {adata_map.uns['train_genes_df']['train_score'].mean():.4f}")

    # ------------------------------------------------------------------
    # 8. Evaluate results
    # ------------------------------------------------------------------
    ct_pred = adata_sp.obsm["plantdeconv_ct_pred"].copy()
    if not isinstance(ct_pred, pd.DataFrame):
        ct_pred = pd.DataFrame(ct_pred, index=adata_sp.obs_names)
    ct_pred_norm = ct_pred.div(ct_pred.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    arr = np.sort(ct_pred_norm.to_numpy(dtype=float), axis=1)[:, ::-1]
    top1 = pd.Series(arr[:, 0], index=ct_pred_norm.index)
    top2 = (
        pd.Series(arr[:, 0] + arr[:, 1], index=ct_pred_norm.index)
        if arr.shape[1] > 1
        else top1.copy()
    )
    nonzero_005 = (ct_pred_norm >= 0.05).sum(axis=1)

    layer_metrics = pdv.compute_layer_consistency_metrics(
        ct_pred=ct_pred_norm,
        spot_meta=spot_meta,
        ct_layer_prior=prior_result.prior,
    )
    log(f"Mean out-of-band fraction: {layer_metrics['out_of_band_fraction'].mean():.4f}")

    # ------------------------------------------------------------------
    # 9. Save outputs
    # ------------------------------------------------------------------
    prefix = "plantdeconv"

    adata_map.write_h5ad(OUTPUT_DIR / f"{prefix}_map.h5ad")
    adata_sp.write_h5ad(OUTPUT_DIR / f"{prefix}_spatial_annotated.h5ad")
    adata_ge.write_h5ad(OUTPUT_DIR / f"{prefix}_projected_genes.h5ad")
    log("H5AD files saved.")

    ct_pred.to_csv(OUTPUT_DIR / f"{prefix}_celltype_scores.csv")
    ct_pred_norm.to_csv(OUTPUT_DIR / f"{prefix}_celltype_proportions.csv")
    df_compare.to_csv(OUTPUT_DIR / f"{prefix}_gene_scores.csv")
    adata_map.uns["train_genes_df"].to_csv(OUTPUT_DIR / f"{prefix}_train_genes.csv")
    training_history_to_frame(adata_map.uns["training_history"]).to_csv(
        OUTPUT_DIR / f"{prefix}_training_history.csv", index=False,
    )

    prior_result.prior.to_csv(OUTPUT_DIR / f"{prefix}_ct_layer_prior.csv")
    prior_result.support.to_csv(OUTPUT_DIR / f"{prefix}_ct_layer_support.csv")
    prior_result.similarity.to_csv(OUTPUT_DIR / f"{prefix}_ct_layer_similarity.csv")
    prior_result.layer_bulk.to_csv(OUTPUT_DIR / f"{prefix}_layer_pseudobulk.csv")
    spot_meta.to_csv(OUTPUT_DIR / f"{prefix}_spot_layers.csv")
    layer_metrics.to_csv(OUTPUT_DIR / f"{prefix}_layer_consistency.csv")

    pd.DataFrame({
        "spot": ct_pred_norm.index,
        "top1_fraction": top1.values,
        "top2_fraction": top2.values,
        "n_types_ge_0.05": nonzero_005.values,
    }).to_csv(OUTPUT_DIR / f"{prefix}_spot_summary.csv", index=False)

    log("All CSV files saved.")

    # ------------------------------------------------------------------
    # 10. Plots
    # ------------------------------------------------------------------
    save_spatial_preview(adata_sp, OUTPUT_DIR)

    try:
        pdv.plot_pie_map(
            adata=adata_sp,
            out_path=OUTPUT_DIR / f"{prefix}_pie_map.png",
            title="PlantDeconv Cell Type Proportions",
        )
        log("Pie chart map saved.")
    except Exception as e:
        log(f"Pie chart plot skipped: {e}")

    try:
        pdv.plot_training_scores(adata_map)
        plt.savefig(OUTPUT_DIR / f"{prefix}_training_diagnostics.png", dpi=200, bbox_inches="tight")
        plt.close()
        log("Training diagnostics plot saved.")
    except Exception as e:
        log(f"Training diagnostics plot skipped: {e}")

    # ------------------------------------------------------------------
    # 11. Summary JSON
    # ------------------------------------------------------------------
    summary = {
        "method": "PlantDeconv",
        "version": pdv.__version__,
        "framework": "differentiation_continuity_aware",
        "cluster_label": CLUSTER_LABEL,
        "layer_key": layer_key,
        "layer_order": layer_order,
        "ordering_strategy": ordering_strategy,
        "training_strategy": "differentiation_continuity_marker_driven",
        "marker_key": MARKER_KEY,
        "n_markers_per_celltype": N_MARKERS_PER_CELLTYPE,
        "marker_min_logfc": MARKER_MIN_LOGFC,
        "marker_max_pval_adj": MARKER_MAX_PVAL_ADJ,
        "n_training_genes": int(len(adata_sc.uns["training_genes"])),
        "n_overlap_genes": int(len(adata_sc.uns["overlap_genes"])),
        "n_cells": int(adata_sc.n_obs),
        "n_spots": int(adata_sp.n_obs),
        "n_cell_types": int(adata_sc.obs[CLUSTER_LABEL].nunique()),
        "n_regions": len(layer_order),
        "density_prior": DENSITY_PRIOR,
        "scale_clusters": SCALE_CLUSTERS,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lambda_r": LAMBDA_R,
        "lambda_spot_entropy": LAMBDA_SPOT_ENTROPY,
        "lambda_ct_islands": LAMBDA_CT_ISLANDS,
        "lambda_neighborhood_g1": LAMBDA_NEIGHBORHOOD_G1,
        "lambda_layer_prior": LAMBDA_LAYER_PRIOR,
        "lambda_out_of_band": LAMBDA_OUT_OF_BAND,
        "lambda_layer_distance": LAMBDA_LAYER_DISTANCE,
        # v2.1 three-layer parameters
        "lambda_spatial_continuity": LAMBDA_SPATIAL_CONTINUITY,
        "lambda_differentiation_gradient": LAMBDA_DIFFERENTIATION_GRADIENT,
        "expression_weight": EXPRESSION_WEIGHT,
        "gradient_sigma": GRADIENT_SIGMA,
        "adaptive_floor": ADAPTIVE_FLOOR,
        "adaptive_ceiling": ADAPTIVE_CEILING,
        "prior_temperature": PRIOR_TEMPERATURE,
        "layer_bandwidth": LAYER_BANDWIDTH,
        "random_state": RANDOM_STATE,
        "device": device,
        "results": {
            "mean_training_score": float(adata_map.uns["train_genes_df"]["train_score"].mean()),
            "mean_top1_fraction": float(top1.mean()),
            "mean_top2_fraction": float(top2.mean()),
            "mean_n_types_ge_0.05": float(nonzero_005.mean()),
            "mean_out_of_band_fraction": float(layer_metrics["out_of_band_fraction"].mean()),
        },
    }
    summary_path = OUTPUT_DIR / f"{prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"Summary JSON saved: {summary_path}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    log("=" * 60)
    log("PlantDeconv v2.1 pipeline completed successfully!")
    log(f"Results directory: {OUTPUT_DIR}")
    log(f"  Ordering strategy:    {ordering_strategy}")
    log(f"  Training score:       {summary['results']['mean_training_score']:.4f}")
    log(f"  Top-1 fraction:       {summary['results']['mean_top1_fraction']:.4f}")
    log(f"  Top-2 fraction:       {summary['results']['mean_top2_fraction']:.4f}")
    log(f"  Types per spot (>=5%): {summary['results']['mean_n_types_ge_0.05']:.2f}")
    log(f"  Out-of-band fraction: {summary['results']['mean_out_of_band_fraction']:.4f}")
    log("=" * 60)


if __name__ == "__main__":
    main()
