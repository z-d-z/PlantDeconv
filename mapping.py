"""
Mapping functions for PlantDeconv v2.1.

High-level functions that orchestrate deconvolution: preparing data tensors,
instantiating the optimizer, running training, and packaging results.

v2.1 simplifies the interface by removing explicit topology selection and
instead using three layers of differentiation-continuity regularisation
that work universally across all plant tissue types.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import csc_matrix, csr_matrix

from . import optimizer as opt
from . import utils as ut
from .preprocessing import adata_to_cluster_expression
from .layer_prior import (
    build_layer_penalty_inputs,
    build_spatial_continuity_inputs,
    build_differentiation_gradient_inputs,
    build_adaptive_regularization_weights,
    extract_spatial_coordinates,
    _to_dense_matrix,
)
from .layer_aware_optimizer import LayerAwareMapper

logging.getLogger().setLevel(logging.INFO)

EPS = 1e-12


def _compute_spatial_weights(adata_sp, standardized, self_inclusion):
    """Compute spatial weights from precomputed neighbourhood or from coordinates."""
    import sklearn.preprocessing as skp

    if {"spatial_connectivities", "spatial_distances"}.issubset(set(adata_sp.obsp.keys())):
        if standardized:
            spatial_w = skp.normalize(
                adata_sp.obsp["spatial_distances"].copy(),
                norm="l1", axis=1, copy=False,
            ).toarray()
        else:
            spatial_w = adata_sp.obsp["spatial_connectivities"].todense()
        spatial_w = np.asarray(spatial_w, dtype=np.float32)
        if self_inclusion:
            spatial_w += np.eye(spatial_w.shape[0], dtype=np.float32)
        return spatial_w

    # Fallback: build from coordinates
    from scipy.spatial import cKDTree
    coords = extract_spatial_coordinates(adata_sp).to_numpy(dtype=np.float32)
    tree = cKDTree(coords)
    n_neighbors = 6
    k = min(n_neighbors + 1, len(coords))
    dists, neighbors = tree.query(coords, k=k)
    if dists.ndim == 1:
        dists = dists[:, None]
        neighbors = neighbors[:, None]

    weights = np.zeros((len(coords), len(coords)), dtype=np.float32)
    finite = dists[:, 1:][np.isfinite(dists[:, 1:]) & (dists[:, 1:] > 0)]
    ref_dist = float(np.median(finite)) if finite.size else 1.0
    ref_dist = max(ref_dist, 1e-3)

    for i in range(len(coords)):
        for dist, j in zip(dists[i, 1:], neighbors[i, 1:]):
            if not np.isfinite(dist):
                continue
            if standardized:
                value = 1.0 / max(float(dist), ref_dist * 0.25)
            else:
                value = 1.0
            weights[i, int(j)] = max(weights[i, int(j)], value)
            weights[int(j), i] = max(weights[int(j), i], value)

    if standardized:
        row_sum = weights.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        weights = weights / row_sum
    if self_inclusion:
        weights = weights + np.eye(weights.shape[0], dtype=np.float32)
    return weights


def _to_dense(adata, genes):
    """Extract dense matrix from AnnData for given genes."""
    if isinstance(adata.X, (csc_matrix, csr_matrix)):
        return np.array(adata[:, genes].X.toarray(), dtype="float32")
    elif isinstance(adata.X, np.ndarray):
        return np.array(adata[:, genes].X, dtype="float32")
    else:
        X_type = type(adata.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError


def map_cells_to_space(
    adata_sc,
    adata_sp,
    cv_train_genes=None,
    cluster_label=None,
    mode="cells",
    device="cpu",
    learning_rate=0.1,
    num_epochs=1000,
    scale=True,
    lambda_d=0,
    lambda_g1=1,
    lambda_g2=0,
    lambda_r=0,
    lambda_l1=0,
    lambda_l2=0,
    lambda_spot_entropy=0,
    lambda_count=1,
    lambda_f_reg=1,
    target_count=None,
    lambda_neighborhood_g1=0,
    lambda_ct_islands=0,
    lambda_getis_ord=0,
    lambda_moran=0,
    lambda_geary=0,
    random_state=None,
    verbose=True,
    density_prior="rna_count_based",
):
    """
    Map single-cell data onto spatial data (base method, no structure awareness).

    Supports three modes:
    - 'cells': map individual cells.
    - 'clusters': map aggregated cluster profiles.
    - 'constrained': map with cell filtering.
    """
    if lambda_g1 == 0:
        raise ValueError("lambda_g1 cannot be 0.")

    if (type(density_prior) is str) and (
        density_prior not in ["rna_count_based", "uniform", None]
    ):
        raise ValueError("Invalid input for density_prior.")

    if density_prior is not None and (lambda_d == 0 or lambda_d is None):
        lambda_d = 1

    if lambda_d > 0 and density_prior is None:
        raise ValueError("When lambda_d is set, please define the density_prior.")

    if mode not in ["clusters", "cells", "constrained"]:
        raise ValueError('Argument "mode" must be "cells", "clusters" or "constrained".')

    if mode == "clusters" and cluster_label is None:
        raise ValueError("A cluster_label must be specified if mode is 'clusters'.")

    if mode == "constrained" and not all([target_count, lambda_f_reg, lambda_count]):
        raise ValueError(
            "target_count, lambda_f_reg and lambda_count must be specified if mode is 'constrained'."
        )

    if mode == "clusters":
        adata_sc = adata_to_cluster_expression(
            adata_sc, cluster_label, scale, add_density=True
        )

    if not {"training_genes", "overlap_genes"}.issubset(set(adata_sc.uns.keys())):
        raise ValueError("Missing preprocessing parameters. Run `pp_adatas()`.")
    if not {"training_genes", "overlap_genes"}.issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing preprocessing parameters. Run `pp_adatas()`.")

    assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

    if cv_train_genes is None:
        training_genes = adata_sc.uns["training_genes"]
    else:
        if set(cv_train_genes).issubset(set(adata_sc.uns["training_genes"])):
            training_genes = cv_train_genes
        else:
            raise ValueError("Given training genes should be a subset of preprocessed genes.")

    logging.info("Allocate tensors for mapping.")
    S = _to_dense(adata_sc, training_genes)
    G = _to_dense(adata_sp, training_genes)

    if not S.any(axis=0).all() or not G.any(axis=0).all():
        raise ValueError("Genes with all zero values detected. Run `pp_adatas()`.")

    d_source = None
    d_str = density_prior
    if type(density_prior) is np.ndarray:
        d_str = "customized"

    if density_prior == "rna_count_based":
        density_prior = adata_sp.obs["rna_count_based_density"]
    elif density_prior == "uniform":
        density_prior = adata_sp.obs["uniform_density"]

    if mode == "cells":
        d = density_prior
    if mode == "clusters":
        d_source = np.array(adata_sc.obs["cluster_density"])
    if mode in ["clusters", "constrained"]:
        if density_prior is None:
            d = adata_sp.obs["uniform_density"]
            d_str = "uniform"
        else:
            d = density_prior
        if lambda_d is None or lambda_d == 0:
            lambda_d = 1

    device = torch.device(device)
    print_each = 100 if verbose else None

    if mode in ["cells", "clusters"]:
        voxel_weights, neighborhood_filter, ct_encode, spatial_weights = None, None, None, None
        if lambda_neighborhood_g1 > 0:
            voxel_weights = _compute_spatial_weights(adata_sp, standardized=True, self_inclusion=True)
        if lambda_ct_islands > 0:
            if cluster_label not in adata_sc.obs.keys():
                raise ValueError("cluster_label must be specified for cell type island extension.")
            neighborhood_filter = _compute_spatial_weights(adata_sp, standardized=False, self_inclusion=False)
            ct_encode = ut.one_hot_encoding(adata_sc.obs[cluster_label]).values
        if lambda_moran > 0 or lambda_geary > 0:
            spatial_weights = _compute_spatial_weights(adata_sp, standardized=True, self_inclusion=False)
        if lambda_getis_ord > 0:
            spatial_weights = _compute_spatial_weights(adata_sp, standardized=False, self_inclusion=True)

        hyperparameters = {
            "lambda_d": lambda_d, "lambda_g1": lambda_g1, "lambda_g2": lambda_g2,
            "lambda_r": lambda_r, "lambda_l1": lambda_l1, "lambda_l2": lambda_l2,
            "lambda_spot_entropy": lambda_spot_entropy,
            "d_source": d_source,
            "lambda_neighborhood_g1": lambda_neighborhood_g1,
            "voxel_weights": voxel_weights,
            "lambda_ct_islands": lambda_ct_islands,
            "neighborhood_filter": neighborhood_filter,
            "ct_encode": ct_encode,
            "lambda_getis_ord": lambda_getis_ord,
            "lambda_moran": lambda_moran,
            "lambda_geary": lambda_geary,
            "spatial_weights": spatial_weights,
        }

        logging.info(
            "Begin training with {} genes and {} density_prior in {} mode...".format(
                len(training_genes), d_str, mode
            )
        )
        mapper = opt.Mapper(
            S=S, G=G, d=d, device=device, random_state=random_state, **hyperparameters,
        )
        mapping_matrix, training_history = mapper.train(
            learning_rate=learning_rate, num_epochs=num_epochs, print_each=print_each,
        )

    elif mode == "constrained":
        hyperparameters = {
            "lambda_d": lambda_d, "lambda_g1": lambda_g1, "lambda_g2": lambda_g2,
            "lambda_r": lambda_r, "lambda_count": lambda_count,
            "lambda_f_reg": lambda_f_reg, "target_count": target_count,
        }
        logging.info(
            "Begin training with {} genes and {} density_prior in {} mode...".format(
                len(training_genes), d_str, mode
            )
        )
        mapper = opt.MapperConstrained(
            S=S, G=G, d=d, device=device, random_state=random_state, **hyperparameters,
        )
        mapping_matrix, F_out, training_history = mapper.train(
            learning_rate=learning_rate, num_epochs=num_epochs, print_each=print_each,
        )

    logging.info("Saving results...")
    adata_map = sc.AnnData(
        X=mapping_matrix,
        obs=adata_sc[:, training_genes].obs.copy(),
        var=adata_sp[:, training_genes].obs.copy(),
    )

    if mode == "constrained":
        adata_map.obs["F_out"] = F_out

    G_predicted = adata_map.X.T @ S
    cos_sims = []
    for v1, v2 in zip(G.T, G_predicted.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    df_cs = pd.DataFrame(cos_sims, training_genes, columns=["train_score"])
    df_cs = df_cs.sort_values(by="train_score", ascending=False)
    adata_map.uns["train_genes_df"] = df_cs

    ut.annotate_gene_sparsity(adata_sc)
    ut.annotate_gene_sparsity(adata_sp)
    adata_map.uns["train_genes_df"]["sparsity_sc"] = adata_sc[:, training_genes].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_sp"] = adata_sp[:, training_genes].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_diff"] = (
        adata_sp[:, training_genes].var.sparsity
        - adata_sc[:, training_genes].var.sparsity
    )
    adata_map.uns["training_history"] = training_history

    return adata_map


def map_clusters_to_space_layeraware(
    adata_sc: sc.AnnData,
    adata_sp: sc.AnnData,
    *,
    cluster_label: str,
    ct_layer_prior: pd.DataFrame,
    spot_meta: pd.DataFrame,
    cv_train_genes: Optional[Sequence[str]] = None,
    device: str = "cpu",
    learning_rate: float = 0.08,
    num_epochs: int = 1500,
    scale: bool = False,
    lambda_d: float = 0.0,
    lambda_g1: float = 1.0,
    lambda_g2: float = 0.0,
    lambda_r: float = 0.0,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
    lambda_spot_entropy: float = 0.0,
    lambda_neighborhood_g1: float = 0.0,
    lambda_ct_islands: float = 0.0,
    lambda_getis_ord: float = 0.0,
    lambda_moran: float = 0.0,
    lambda_geary: float = 0.0,
    # ---- Region-based penalties (optional) ----
    lambda_layer_prior: float = 0.0,
    lambda_out_of_band: float = 0.0,
    lambda_layer_distance: float = 0.0,
    # ---- Differentiation-continuity penalties (core v2.1) ----
    lambda_spatial_continuity: float = 0.0,
    lambda_differentiation_gradient: float = 0.0,
    continuity_neighbors: int = 6,
    expression_weight: float = 0.5,
    gradient_sigma: float = 2.0,
    adaptive_floor: float = 0.1,
    adaptive_ceiling: float = 1.0,
    density_prior: Optional[object] = "uniform",
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> sc.AnnData:
    """
    Map cluster-level expression to space with differentiation-continuity
    regularisation.

    This is the core PlantDeconv v2.1 function.  It extends standard
    cluster-mode mapping with three layers of differentiation-continuity
    penalties that capture universal plant tissue properties — no topology
    selection required.

    Parameters
    ----------
    adata_sc : AnnData
        Single-cell data.
    adata_sp : AnnData
        Spatial data.
    cluster_label : str
        Cell type annotation column.
    ct_layer_prior : DataFrame
        Cell-type-to-region prior (from ``build_ct_layer_prior``).
    spot_meta : DataFrame
        Spot metadata (from ``build_spot_layer_metadata``).
    lambda_spatial_continuity : float
        Weight for Layer 1 — expression-aware spatial continuity.
        Recommended range: 0.1–1.0.  Default 0.0 (disabled).
    lambda_differentiation_gradient : float
        Weight for Layer 2 — anisotropic differentiation gradient.
        Recommended range: 0.1–0.5.  Default 0.0 (disabled).
    continuity_neighbors : int
        Number of spatial neighbours for continuity/gradient construction.
    expression_weight : float
        Layer 1: balance between spatial and expression weighting.
        0 = pure spatial, 1 = pure expression.  Default 0.5.
    gradient_sigma : float
        Layer 2: bandwidth of differentiation distance kernel.
        Larger → smoother gradient.  Default 2.0.
    adaptive_floor : float
        Layer 3: minimum regularisation weight at boundaries.
    adaptive_ceiling : float
        Layer 3: maximum regularisation weight in homogeneous areas.

    Returns
    -------
    AnnData
        Cluster-by-spot mapping result.
    """
    if cluster_label not in adata_sc.obs.columns:
        raise KeyError(f"Missing adata_sc.obs['{cluster_label}'].")

    if not {"training_genes", "overlap_genes"}.issubset(adata_sc.uns.keys()):
        raise ValueError("Missing preprocessing keys in adata_sc. Run pp_adatas first.")
    if not {"training_genes", "overlap_genes"}.issubset(adata_sp.uns.keys()):
        raise ValueError("Missing preprocessing keys in adata_sp. Run pp_adatas first.")

    if cv_train_genes is None:
        training_genes = list(adata_sc.uns["training_genes"])
    else:
        training_genes = list(cv_train_genes)
        if not set(training_genes).issubset(set(adata_sc.uns["training_genes"])):
            raise ValueError("cv_train_genes must be a subset of preprocessing training genes.")

    adata_sc_cluster = adata_to_cluster_expression(
        adata_sc, cluster_label=cluster_label, scale=scale, add_density=True,
    )
    cluster_names = adata_sc_cluster.obs[cluster_label].astype(str).tolist()
    ct_layer_prior = ct_layer_prior.copy()
    ct_layer_prior.index = ct_layer_prior.index.astype(str)
    ct_layer_prior = ct_layer_prior.loc[cluster_names]

    spot_meta = spot_meta.loc[adata_sp.obs_names].copy()
    layer_order = ct_layer_prior.columns.astype(str).tolist()

    # ---- Build region-based penalty inputs (optional) ----
    spot_layer_matrix, out_of_band_mask, ct_spot_distance = build_layer_penalty_inputs(
        spot_meta=spot_meta,
        layer_order=layer_order,
        ct_layer_prior=ct_layer_prior,
    )

    # ---- Build differentiation-continuity inputs (core v2.1) ----

    # Layer 1: Expression-aware spatial adjacency
    spot_adjacency = None
    if lambda_spatial_continuity > 0:
        logging.info("Building Layer 1: expression-aware spatial adjacency...")
        spot_adjacency = build_spatial_continuity_inputs(
            adata_sp, spot_meta,
            n_neighbors=continuity_neighbors,
            expression_weight=expression_weight,
        )

    # Layer 2: Anisotropic differentiation gradient weights
    gradient_weights = None
    if lambda_differentiation_gradient > 0:
        logging.info("Building Layer 2: differentiation gradient weights...")
        gradient_weights = build_differentiation_gradient_inputs(
            adata_sc, adata_sp, spot_meta, layer_order,
            cluster_label=cluster_label,
            n_neighbors=continuity_neighbors,
            sigma=gradient_sigma,
        )

    # Layer 3: Adaptive per-spot regularisation
    adaptive_weights = None
    if lambda_spatial_continuity > 0 or lambda_differentiation_gradient > 0:
        logging.info("Building Layer 3: adaptive per-spot regularisation weights...")
        adaptive_weights = build_adaptive_regularization_weights(
            adata_sp, spot_meta,
            n_neighbors=continuity_neighbors,
            floor=adaptive_floor,
            ceiling=adaptive_ceiling,
        )

    S = _to_dense_matrix(adata_sc_cluster, training_genes)
    G = _to_dense_matrix(adata_sp, training_genes)

    d_source = np.asarray(adata_sc_cluster.obs["cluster_density"], dtype=np.float32)
    if isinstance(density_prior, str):
        if density_prior == "rna_count_based":
            d = np.asarray(adata_sp.obs["rna_count_based_density"], dtype=np.float32)
        elif density_prior == "uniform":
            d = np.asarray(adata_sp.obs["uniform_density"], dtype=np.float32)
        elif density_prior == "none":
            d = None
        else:
            raise ValueError(f"Invalid density_prior: {density_prior}")
    elif density_prior is None:
        d = None
    else:
        d = np.asarray(density_prior, dtype=np.float32)

    if d is not None and (lambda_d is None or lambda_d == 0):
        lambda_d = 1.0
    if d is None:
        lambda_d = 0.0

    voxel_weights = None
    if lambda_neighborhood_g1 > 0:
        voxel_weights = _compute_spatial_weights(adata_sp, standardized=True, self_inclusion=True)

    neighborhood_filter = None
    ct_encode = None
    if lambda_ct_islands > 0:
        neighborhood_filter = _compute_spatial_weights(adata_sp, standardized=False, self_inclusion=False)
        ct_encode = ut.one_hot_encoding(adata_sc_cluster.obs[cluster_label]).values

    spatial_weights = None
    if lambda_moran > 0 or lambda_geary > 0:
        spatial_weights = _compute_spatial_weights(adata_sp, standardized=True, self_inclusion=False)
    if lambda_getis_ord > 0:
        spatial_weights = _compute_spatial_weights(adata_sp, standardized=False, self_inclusion=True)

    mapper = LayerAwareMapper(
        S=S, G=G, d=d, d_source=d_source,
        lambda_d=lambda_d, lambda_g1=lambda_g1, lambda_g2=lambda_g2,
        lambda_r=lambda_r, lambda_l1=lambda_l1, lambda_l2=lambda_l2,
        lambda_spot_entropy=lambda_spot_entropy,
        lambda_neighborhood_g1=lambda_neighborhood_g1,
        voxel_weights=voxel_weights,
        lambda_getis_ord=lambda_getis_ord,
        lambda_geary=lambda_geary,
        lambda_moran=lambda_moran,
        neighborhood_filter=neighborhood_filter,
        ct_encode=ct_encode,
        lambda_ct_islands=lambda_ct_islands,
        spatial_weights=spatial_weights,
        # Region-based (optional)
        spot_layer_matrix=spot_layer_matrix,
        ct_layer_prior=ct_layer_prior.to_numpy(dtype=np.float32),
        out_of_band_mask=out_of_band_mask,
        ct_spot_distance=ct_spot_distance,
        lambda_layer_prior=lambda_layer_prior,
        lambda_out_of_band=lambda_out_of_band,
        lambda_layer_distance=lambda_layer_distance,
        # Differentiation-continuity (core v2.1)
        spot_adjacency=spot_adjacency,
        gradient_weights=gradient_weights,
        adaptive_weights=adaptive_weights,
        lambda_spatial_continuity=lambda_spatial_continuity,
        lambda_differentiation_gradient=lambda_differentiation_gradient,
        device=device,
        random_state=random_state,
    )

    print_each = 100 if verbose else None
    mapping_matrix, training_history = mapper.train(
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        print_each=print_each,
    )

    adata_map = sc.AnnData(
        X=mapping_matrix,
        obs=adata_sc_cluster[:, training_genes].obs.copy(),
        var=adata_sp[:, training_genes].obs.copy(),
    )

    G_predicted = adata_map.X.T @ S
    cos_sims = []
    for v1, v2 in zip(G.T, G_predicted.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / max(norm_sq, EPS))

    df_cs = pd.DataFrame(cos_sims, training_genes, columns=["train_score"])
    df_cs = df_cs.sort_values(by="train_score", ascending=False)
    adata_map.uns["train_genes_df"] = df_cs

    ut.annotate_gene_sparsity(adata_sc_cluster)
    ut.annotate_gene_sparsity(adata_sp)
    adata_map.uns["train_genes_df"]["sparsity_sc"] = adata_sc_cluster[:, training_genes].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_sp"] = adata_sp[:, training_genes].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_diff"] = (
        adata_sp[:, training_genes].var.sparsity
        - adata_sc_cluster[:, training_genes].var.sparsity
    )
    adata_map.uns["training_history"] = training_history
    adata_map.uns["layer_order"] = list(layer_order)
    adata_map.uns["ct_layer_prior"] = ct_layer_prior
    adata_map.uns["spot_layer_meta"] = spot_meta.loc[adata_sp.obs_names]
    adata_map.uns["ordering_strategy"] = spot_meta.attrs.get("ordering_strategy", "auto")

    return adata_map
