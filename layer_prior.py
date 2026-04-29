"""
Spatial prior construction for PlantDeconv v2.1.

Builds differentiation-continuity-aware metadata and cell-type-to-region
priors from spatial transcriptomics data.  The core innovation of v2.1 is
modelling the **spatial continuity of cell differentiation** — a universal
property of plant tissues — as regularisation constraints, without
requiring any geometric topology assumptions.

Three layers of differentiation-continuity regularisation:

* **Layer 1 — Expression-aware adaptive spatial neighbourhood**:
  Edge weights combine spatial proximity with expression similarity,
  so that biological boundaries (large expression differences) naturally
  receive less smoothing.

* **Layer 2 — Anisotropic differentiation gradient smoothing**:
  Along differentiation trajectories, gradual composition change is
  expected; perpendicular to the trajectory, composition should be
  uniform.  Neighbour weights are modulated by cell-type differentiation
  distance derived from reference scRNA-seq data.

* **Layer 3 — Adaptive per-spot regularisation strength**:
  Homogeneous regions (e.g., xylem interior) receive stronger smoothing;
  heterogeneous boundary regions (e.g., cambium zone) receive weaker
  smoothing.  Strength is determined automatically from local expression
  heterogeneity.

Region ordering is handled internally via automatic heuristics and is
transparent to the user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix
from scipy.spatial import cKDTree

from .preprocessing import adata_to_cluster_expression

LOGGER = logging.getLogger(__name__)
EPS = 1e-12


# =====================================================================
# Result container
# =====================================================================

@dataclass
class LayerPriorResult:
    """Container returned by :func:`build_ct_layer_prior`.

    Kept as *LayerPriorResult* for backward compatibility; the semantics
    have been generalised from "layer" to "spatial region".
    """
    prior: pd.DataFrame
    support: pd.DataFrame
    similarity: pd.DataFrame
    layer_bulk: pd.DataFrame          # region-level pseudobulk
    spot_meta: pd.DataFrame
    layer_order: List[str]            # ordered region labels
    layer_key: str
    region_adjacency: Optional[np.ndarray] = field(default=None, repr=False)

# Alias for new code
SpatialPriorResult = LayerPriorResult


# =====================================================================
# Internal helpers
# =====================================================================

def _to_dense_matrix(adata: sc.AnnData, genes: Sequence[str]) -> np.ndarray:
    matrix = adata[:, list(genes)].X
    if isinstance(matrix, (csr_matrix, csc_matrix)):
        return np.asarray(matrix.toarray(), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _row_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scale = max(float(temperature), 1e-4)
    shifted = values / scale
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.clip(exp_values.sum(axis=1, keepdims=True), EPS, None)


def _pairwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    denom = np.clip(left_norm * right_norm.T, EPS, None)
    return (left @ right.T) / denom


def _mean_by_group(
    matrix: np.ndarray,
    groups: Sequence[str],
    ordered_groups: Sequence[str],
) -> np.ndarray:
    grouped_rows = []
    group_array = np.asarray(list(groups))
    for label in ordered_groups:
        mask = group_array == str(label)
        if not np.any(mask):
            grouped_rows.append(np.zeros(matrix.shape[1], dtype=np.float32))
            continue
        grouped_rows.append(matrix[mask].mean(axis=0))
    return np.asarray(grouped_rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Section / connected-component detection
# ---------------------------------------------------------------------------

def _connected_components_from_knn(
    coords: np.ndarray,
    n_neighbors: int = 6,
    threshold_factor: float = 2.5,
) -> np.ndarray:
    """Identify tissue sections via connected-component analysis on KNN graph."""
    n_spots = coords.shape[0]
    if n_spots == 0:
        return np.array([], dtype=int)
    if n_spots == 1:
        return np.zeros(1, dtype=int)

    tree = cKDTree(coords)
    k = min(n_neighbors + 1, n_spots)
    dists, neighbors = tree.query(coords, k=k)
    if dists.ndim == 1:
        dists = dists[:, None]
        neighbors = neighbors[:, None]

    nn_dists = dists[:, 1:].reshape(-1)
    nn_dists = nn_dists[np.isfinite(nn_dists) & (nn_dists > 0)]
    if nn_dists.size == 0:
        return np.zeros(n_spots, dtype=int)
    connect_threshold = float(np.median(nn_dists) * threshold_factor)

    parent = np.arange(n_spots, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(n_spots):
        for dist, j in zip(dists[i, 1:], neighbors[i, 1:]):
            if not np.isfinite(dist):
                continue
            if dist <= connect_threshold:
                union(i, int(j))

    roots = np.array([find(i) for i in range(n_spots)], dtype=int)
    unique_roots = {root: idx for idx, root in enumerate(pd.unique(roots))}
    return np.array([unique_roots[root] for root in roots], dtype=int)


# ---------------------------------------------------------------------------
# Region adjacency from spatial KNN
# ---------------------------------------------------------------------------

def _build_region_adjacency(
    coords: np.ndarray,
    region_labels: np.ndarray,
    ordered_regions: Sequence[str],
    n_neighbors: int = 10,
) -> np.ndarray:
    """Build a region-adjacency matrix from the spatial KNN graph.

    Returns
    -------
    adj : ndarray, shape (n_regions, n_regions)
        Symmetric matrix where adj[i, j] = fraction of cross-boundary
        KNN edges between region *i* and region *j*, normalised so that
        each row sums to 1.
    """
    n_regions = len(ordered_regions)
    region_to_idx = {str(r): i for i, r in enumerate(ordered_regions)}
    label_idx = np.array([region_to_idx.get(str(l), -1) for l in region_labels])

    tree = cKDTree(coords)
    k = min(n_neighbors + 1, len(coords))
    _, neighbors = tree.query(coords, k=k)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]

    adj = np.zeros((n_regions, n_regions), dtype=np.float64)
    for i in range(len(coords)):
        ri = label_idx[i]
        if ri < 0:
            continue
        for j_idx in neighbors[i, 1:]:
            rj = label_idx[int(j_idx)]
            if rj < 0:
                continue
            adj[ri, rj] += 1.0
            adj[rj, ri] += 1.0

    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    adj = adj / row_sum
    return adj.astype(np.float32)


# ---------------------------------------------------------------------------
# Automatic region ordering (internal — not exposed to user)
# ---------------------------------------------------------------------------

def _compute_section_centers(
    meta: pd.DataFrame,
    layer_order: Sequence[str],
) -> Dict[int, Tuple[float, float]]:
    centers: Dict[int, Tuple[float, float]] = {}
    anchor_layers = list(layer_order[:2]) if len(layer_order) > 1 else list(layer_order)
    for section_id, df_section in meta.groupby("section_id"):
        anchor = df_section[df_section["layer_label"].isin(anchor_layers)]
        if anchor.empty:
            anchor = df_section
        centers[int(section_id)] = (
            float(anchor["x"].mean()),
            float(anchor["y"].mean()),
        )
    return centers


def _attach_radius(
    meta: pd.DataFrame,
    centers: Dict[int, Tuple[float, float]],
) -> pd.DataFrame:
    meta = meta.copy()
    radius_values = np.zeros(len(meta), dtype=np.float32)
    positions = np.arange(len(meta))
    for section_id, df_section in meta.groupby("section_id"):
        cx, cy = centers[int(section_id)]
        dx = df_section["x"].to_numpy(dtype=np.float32) - cx
        dy = df_section["y"].to_numpy(dtype=np.float32) - cy
        radius = np.sqrt(dx ** 2 + dy ** 2)
        if np.ptp(radius) > 0:
            radius = (radius - radius.min()) / (radius.max() - radius.min())
        else:
            radius = np.zeros_like(radius)
        radius_values[positions[df_section.index.map(meta.index.get_loc)]] = radius
    meta["radius_norm"] = radius_values
    return meta


def _order_regions_by_graph_traversal(
    region_adjacency: np.ndarray,
    ordered_regions: List[str],
) -> List[str]:
    """Order regions by greedy graph traversal starting from the most
    central region (highest self-connection ratio)."""
    n = len(ordered_regions)
    if n <= 1:
        return list(ordered_regions)

    diag = np.diag(region_adjacency)
    start = int(np.argmax(diag))

    visited = [False] * n
    order = []
    current = start
    for _ in range(n):
        visited[current] = True
        order.append(ordered_regions[current])
        row = region_adjacency[current].copy()
        row[visited] = -1
        next_node = int(np.argmax(row))
        if row[next_node] <= 0:
            unvisited = [i for i in range(n) if not visited[i]]
            if not unvisited:
                break
            next_node = unvisited[0]
        current = next_node

    return order


def _order_regions_by_axis(
    meta: pd.DataFrame,
    axis: Optional[np.ndarray] = None,
) -> Tuple[List[str], np.ndarray]:
    """Order regions by projection onto a principal axis."""
    coords = meta[["x", "y"]].to_numpy(dtype=np.float64)
    if axis is None:
        centered = coords - coords.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
    axis = axis / (np.linalg.norm(axis) + EPS)

    projections = coords @ axis
    meta = meta.copy()
    meta["_axis_proj"] = projections

    region_mean = (
        meta.groupby("layer_label")["_axis_proj"]
        .mean()
        .sort_values()
    )
    return region_mean.index.astype(str).tolist(), axis.astype(np.float32)


def _auto_detect_and_order(meta: pd.DataFrame, adjacency_neighbors: int = 10) -> Tuple[str, List[str]]:
    """Automatically detect spatial layout and order regions.

    Uses heuristics to determine the best ordering strategy:
    - Aspect ratio ≈ 1 and ≥ 3 regions → concentric (radial ordering)
    - High aspect ratio → linear (axis projection ordering)
    - Otherwise → graph (adjacency traversal ordering)

    Returns
    -------
    detected_strategy : str
        One of "concentric", "linear", "graph" (for logging only).
    layer_order : list of str
        Ordered region labels.
    """
    coords = meta[["x", "y"]].to_numpy(dtype=np.float64)
    ptp = np.ptp(coords, axis=0)
    n_regions = meta["layer_label"].nunique()
    unique_regions = sorted(meta["layer_label"].unique(), key=str)

    if ptp.min() < EPS:
        strategy = "graph"
    else:
        aspect = ptp.max() / ptp.min()
        if aspect < 2.0 and n_regions >= 3:
            strategy = "concentric"
        elif aspect >= 3.0:
            strategy = "linear"
        else:
            strategy = "graph"

    if strategy == "concentric":
        layer_order = _order_concentric(meta)
    elif strategy == "linear":
        layer_order, _ = _order_regions_by_axis(meta)
    else:
        adj = _build_region_adjacency(
            meta[["x", "y"]].to_numpy(dtype=np.float32),
            meta["layer_label"].to_numpy(),
            unique_regions,
            n_neighbors=adjacency_neighbors,
        )
        layer_order = _order_regions_by_graph_traversal(adj, unique_regions)

    return strategy, layer_order


def _order_concentric(meta: pd.DataFrame) -> List[str]:
    """Order regions by mean normalised radius."""
    coarse_centers = {
        int(sec): (float(df["x"].mean()), float(df["y"].mean()))
        for sec, df in meta.groupby("section_id")
    }
    meta = _attach_radius(meta, coarse_centers)
    layer_order = (
        meta.groupby("layer_label")["radius_norm"]
        .mean()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    refined_centers = _compute_section_centers(meta, layer_order)
    meta = _attach_radius(meta, refined_centers)
    layer_order = (
        meta.groupby("layer_label")["radius_norm"]
        .mean()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    return layer_order


# ---------------------------------------------------------------------------
# Marker gene helpers
# ---------------------------------------------------------------------------

def _normalize_marker_dict(
    markers_by_ct: Optional[Dict[str, Sequence[str]]],
    valid_genes: Iterable[str],
) -> Dict[str, List[str]]:
    if not markers_by_ct:
        return {}
    valid_gene_set = {str(g) for g in valid_genes}
    normalized: Dict[str, List[str]] = {}
    for ct, genes in markers_by_ct.items():
        normalized[str(ct)] = [
            str(g).lower() for g in genes if str(g).lower() in valid_gene_set
        ]
    return normalized


# =====================================================================
# Public API — coordinate & metadata extraction
# =====================================================================

def extract_spatial_coordinates(adata_sp: sc.AnnData) -> pd.DataFrame:
    """Extract spatial coordinates from AnnData as a DataFrame with columns ['x', 'y']."""
    if "spatial" in adata_sp.obsm:
        coords = np.asarray(adata_sp.obsm["spatial"])[:, :2]
        return pd.DataFrame(coords, index=adata_sp.obs_names, columns=["x", "y"])
    if {"coor_X", "coor_Y"}.issubset(adata_sp.obs.columns):
        return adata_sp.obs[["coor_X", "coor_Y"]].rename(
            columns={"coor_X": "x", "coor_Y": "y"}
        )
    raise KeyError("No spatial coordinates found in adata_sp.")


def choose_layer_key(
    adata_sp: sc.AnnData,
    layer_key_candidates: Sequence[str] = ("domain", "cluster"),
) -> str:
    """Choose the first available layer annotation key from candidates."""
    for key in layer_key_candidates:
        if key in adata_sp.obs.columns:
            return key
    raise KeyError(
        f"None of the layer key candidates found in adata_sp.obs: {layer_key_candidates}"
    )


# =====================================================================
# Public API — build_spot_layer_metadata  (v2.1: unified, no topology param)
# =====================================================================

def build_spot_layer_metadata(
    adata_sp: sc.AnnData,
    layer_key_candidates: Sequence[str] = ("domain", "cluster"),
    section_threshold_factor: float = 2.5,
    section_neighbors: int = 6,
    region_order: Optional[Sequence[str]] = None,
    adjacency_neighbors: int = 10,
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Build per-spot spatial-region metadata for the spatial data.

    v2.1 removes the explicit ``topology`` parameter.  Region ordering is
    determined automatically via spatial layout heuristics, making the
    framework truly structure-agnostic from the user's perspective.

    Parameters
    ----------
    adata_sp : AnnData
        Spatial data with coordinates in ``obsm['spatial']``.
    layer_key_candidates : sequence of str
        Column names to try as region annotations.
    section_threshold_factor : float
        Distance threshold factor for section detection.
    section_neighbors : int
        KNN neighbours for section detection.
    region_order : sequence of str, optional
        Explicit region ordering.  If provided the automatic ordering
        logic is skipped.
    adjacency_neighbors : int
        KNN neighbours for building region adjacency.

    Returns
    -------
    spot_meta : DataFrame
    layer_order : list of str
    layer_key : str
    """
    coords_df = extract_spatial_coordinates(adata_sp)
    layer_key = choose_layer_key(adata_sp, layer_key_candidates)

    section_id = _connected_components_from_knn(
        coords_df.to_numpy(dtype=np.float32),
        n_neighbors=section_neighbors,
        threshold_factor=section_threshold_factor,
    )
    meta = coords_df.copy()
    meta["section_id"] = section_id
    meta["layer_label"] = adata_sp.obs[layer_key].astype(str).reindex(meta.index)

    # ----- Automatic region ordering -----
    if region_order is not None:
        layer_order = [str(r) for r in region_order]
        detected_strategy = "user_specified"
    else:
        detected_strategy, layer_order = _auto_detect_and_order(
            meta, adjacency_neighbors=adjacency_neighbors,
        )

    LOGGER.info("Region ordering strategy: %s", detected_strategy)

    # ----- Attach normalised positional features -----
    rank_map = {label: rank for rank, label in enumerate(layer_order)}
    denom = max(len(layer_order) - 1, 1)
    meta["layer_rank"] = meta["layer_label"].map(rank_map).fillna(0).astype(int)
    meta["layer_rank_norm"] = meta["layer_rank"].astype(np.float32) / denom

    # Attach radius (useful for diagnostics regardless of layout)
    if detected_strategy == "concentric":
        coarse_centers = {
            int(sec): (float(df["x"].mean()), float(df["y"].mean()))
            for sec, df in meta.groupby("section_id")
        }
        meta = _attach_radius(meta, coarse_centers)
        refined_centers = _compute_section_centers(meta, layer_order)
        meta = _attach_radius(meta, refined_centers)
    else:
        meta["radius_norm"] = meta["layer_rank_norm"]

    meta.attrs["layer_key"] = layer_key
    meta.attrs["ordering_strategy"] = detected_strategy

    return meta, layer_order, layer_key


# =====================================================================
# Public API — pseudobulk construction
# =====================================================================

def build_layer_pseudobulk(
    adata_sp: sc.AnnData,
    genes: Sequence[str],
    spot_meta: pd.DataFrame,
    layer_order: Sequence[str],
) -> pd.DataFrame:
    """Compute mean expression per region from spatial data."""
    x = _to_dense_matrix(adata_sp, genes)
    grouped = _mean_by_group(x, spot_meta["layer_label"].astype(str), layer_order)
    return pd.DataFrame(grouped, index=list(layer_order), columns=list(genes))


def build_cluster_pseudobulk(
    adata_sc: sc.AnnData,
    cluster_label: str,
    genes: Sequence[str],
) -> pd.DataFrame:
    """Compute mean expression per cluster from single-cell data."""
    adata_cluster = adata_to_cluster_expression(
        adata_sc, cluster_label=cluster_label, scale=False, add_density=False,
    )
    cluster_names = adata_cluster.obs[cluster_label].astype(str).tolist()
    matrix = _to_dense_matrix(adata_cluster, genes)
    return pd.DataFrame(matrix, index=cluster_names, columns=list(genes))


# =====================================================================
# Public API — cell-type-to-region prior
# =====================================================================

def build_ct_layer_prior(
    adata_sc: sc.AnnData,
    adata_sp: sc.AnnData,
    cluster_label: str,
    genes: Sequence[str],
    spot_meta: pd.DataFrame,
    layer_order: Sequence[str],
    markers_by_ct: Optional[Dict[str, Sequence[str]]] = None,
    prior_temperature: float = 0.30,
    layer_bandwidth: int = 1,
    min_marker_genes: int = 12,
    marker_boost_weight: float = 0.15,
    region_adjacency: Optional[np.ndarray] = None,
    adjacency_neighbors: int = 10,
) -> LayerPriorResult:
    """
    Build cell-type-to-region prior matrix.

    For each cell type, computes cosine similarity between its pseudobulk
    expression profile and each region's pseudobulk.  The resulting similarity
    matrix is converted to a soft prior via temperature-scaled softmax.

    Support masking uses a k-hop neighbourhood in the region adjacency
    graph (where *k* = ``layer_bandwidth``).  This is a unified approach
    that works for any spatial layout.

    Returns
    -------
    LayerPriorResult
    """
    genes = [str(g) for g in genes]

    layer_bulk = build_layer_pseudobulk(adata_sp, genes, spot_meta, layer_order)
    cluster_bulk = build_cluster_pseudobulk(adata_sc, cluster_label, genes)
    markers_by_ct = _normalize_marker_dict(markers_by_ct, genes)

    ct_names = cluster_bulk.index.astype(str).tolist()
    similarity_rows: List[np.ndarray] = []
    support_rows: List[np.ndarray] = []

    # Build region adjacency for support mask computation
    if region_adjacency is None:
        coords = extract_spatial_coordinates(adata_sp).to_numpy(dtype=np.float32)
        labels = spot_meta["layer_label"].to_numpy()
        region_adjacency = _build_region_adjacency(
            coords, labels, list(layer_order),
            n_neighbors=adjacency_neighbors,
        )

    for ct_name in ct_names:
        marker_genes = markers_by_ct.get(ct_name, [])
        if len(marker_genes) < min_marker_genes:
            marker_genes = genes

        sc_vec = cluster_bulk.loc[ct_name, marker_genes].to_numpy(dtype=np.float32)[None, :]
        layer_mat = layer_bulk.loc[:, marker_genes].to_numpy(dtype=np.float32)
        score = _pairwise_cosine(sc_vec, layer_mat)[0]

        layer_marker_mean = layer_mat.mean(axis=1)
        layer_marker_std = layer_marker_mean.std()
        if layer_marker_std > 0:
            marker_boost = (layer_marker_mean - layer_marker_mean.mean()) / layer_marker_std
        else:
            marker_boost = np.zeros_like(layer_marker_mean)
        score = score + marker_boost_weight * marker_boost.astype(np.float32)
        similarity_rows.append(score.astype(np.float32))

    similarity = np.asarray(similarity_rows, dtype=np.float32)
    prior = _row_softmax(similarity, temperature=prior_temperature)

    for idx in range(prior.shape[0]):
        peak = int(np.argmax(prior[idx]))
        support = _compute_support_mask(
            peak=peak,
            n_regions=prior.shape[1],
            bandwidth=int(layer_bandwidth),
            region_adjacency=region_adjacency,
        )
        prior[idx] = prior[idx] * support
        prior[idx] = prior[idx] / np.clip(prior[idx].sum(), EPS, None)
        support_rows.append(support)

    prior_df = pd.DataFrame(prior, index=ct_names, columns=list(layer_order))
    support_df = pd.DataFrame(
        np.asarray(support_rows, dtype=np.float32),
        index=ct_names, columns=list(layer_order),
    )
    similarity_df = pd.DataFrame(
        similarity, index=ct_names, columns=list(layer_order),
    )

    return LayerPriorResult(
        prior=prior_df,
        support=support_df,
        similarity=similarity_df,
        layer_bulk=layer_bulk,
        spot_meta=spot_meta.copy(),
        layer_order=list(layer_order),
        layer_key=str(spot_meta.attrs.get("layer_key", "")),
        region_adjacency=region_adjacency,
    )


def _compute_support_mask(
    peak: int,
    n_regions: int,
    bandwidth: int,
    region_adjacency: Optional[np.ndarray],
) -> np.ndarray:
    """Compute binary support mask for a cell type.

    Uses k-hop BFS on the region adjacency graph (unified approach).
    Falls back to contiguous window if no adjacency is available.
    """
    support = np.zeros(n_regions, dtype=np.float32)

    if region_adjacency is not None:
        # k-hop BFS on the region adjacency
        visited = {peak}
        frontier = {peak}
        for _ in range(bandwidth):
            next_frontier = set()
            for node in frontier:
                neighbors = np.flatnonzero(region_adjacency[node] > 0)
                for nb in neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier
        for v in visited:
            support[v] = 1.0
    else:
        # Fallback: contiguous window
        lo = max(0, peak - bandwidth)
        hi = min(n_regions, peak + bandwidth + 1)
        support[lo:hi] = 1.0

    return support


# =====================================================================
# Public API — penalty inputs for the optimizer (topology-specific, optional)
# =====================================================================

def build_layer_penalty_inputs(
    spot_meta: pd.DataFrame,
    layer_order: Sequence[str],
    ct_layer_prior: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build tensor inputs for region-based penalty terms (optional in v2.1).

    These penalties are still useful when region annotations are available,
    but the core differentiation-continuity penalties do not require them.

    Returns
    -------
    spot_layer_matrix : ndarray, shape (n_spots, n_regions)
    out_of_band_mask : ndarray, shape (n_ct, n_spots)
    distance_penalty : ndarray, shape (n_ct, n_spots)
    """
    spot_meta = spot_meta.loc[spot_meta.index].copy()
    n_spots = len(spot_meta)
    n_layers = len(layer_order)
    layer_rank_spot = spot_meta["layer_rank"].to_numpy(dtype=int)

    spot_layer_matrix = np.zeros((n_spots, n_layers), dtype=np.float32)
    spot_layer_matrix[np.arange(n_spots), layer_rank_spot] = 1.0

    support = (ct_layer_prior.to_numpy(dtype=np.float32) > 0).astype(np.float32)
    ct_prior_array = ct_layer_prior.to_numpy(dtype=np.float32)
    n_ct = support.shape[0]
    out_of_band = np.zeros((n_ct, n_spots), dtype=np.float32)
    distance_penalty = np.zeros((n_ct, n_spots), dtype=np.float32)
    denom = max(n_layers - 1, 1)

    for i in range(n_ct):
        support_layers = np.flatnonzero(support[i] > 0)
        if support_layers.size == 0:
            support_layers = np.array([int(np.argmax(ct_prior_array[i]))])
        layer_distance = np.abs(layer_rank_spot[None, :] - support_layers[:, None])
        layer_distance = layer_distance.min(axis=0).astype(np.float32)
        out_of_band[i] = (layer_distance > 0).astype(np.float32)
        distance_penalty[i] = layer_distance / denom

    return spot_layer_matrix, out_of_band, distance_penalty


# =====================================================================
# Public API — Layer 1: Expression-aware adaptive spatial neighbourhood
# =====================================================================

def build_spatial_continuity_inputs(
    adata_sp: sc.AnnData,
    spot_meta: pd.DataFrame,
    n_neighbors: int = 6,
    expression_weight: float = 0.5,
    bandwidth_percentile: float = 75.0,
) -> np.ndarray:
    """Build an expression-aware spatial adjacency matrix (Layer 1).

    Unlike v2.0's binary KNN adjacency, this version jointly weights
    edges by spatial proximity and expression similarity.  At biological
    boundaries (where expression profiles change sharply), edge weights
    naturally decay, preventing over-smoothing across true tissue
    boundaries.

    Parameters
    ----------
    adata_sp : AnnData
        Spatial data.
    spot_meta : DataFrame
        Spot metadata (used for index alignment).
    n_neighbors : int
        Number of spatial neighbours per spot.
    expression_weight : float
        Balance between spatial distance and expression similarity.
        0 = pure spatial, 1 = pure expression.  Default 0.5.
    bandwidth_percentile : float
        Percentile of spatial distances used as Gaussian kernel bandwidth.

    Returns
    -------
    spot_adjacency : ndarray, shape (n_spots, n_spots)
        Row-normalised expression-aware spatial adjacency matrix.
    """
    coords = extract_spatial_coordinates(adata_sp).loc[spot_meta.index]
    coord_arr = coords.to_numpy(dtype=np.float32)
    n = len(coord_arr)

    # --- Spatial KNN ---
    tree = cKDTree(coord_arr)
    k = min(n_neighbors + 1, n)
    dists, neighbors = tree.query(coord_arr, k=k)
    if dists.ndim == 1:
        dists = dists[:, None]
        neighbors = neighbors[:, None]

    # --- Spatial distance kernel ---
    finite_dists = dists[:, 1:].reshape(-1)
    finite_dists = finite_dists[np.isfinite(finite_dists) & (finite_dists > 0)]
    if finite_dists.size == 0:
        bandwidth = 1.0
    else:
        bandwidth = float(np.percentile(finite_dists, bandwidth_percentile))
    bandwidth = max(bandwidth, EPS)

    # --- Expression similarity ---
    # Use a subset of highly variable genes for efficiency
    X_sp = adata_sp[spot_meta.index].X
    if isinstance(X_sp, (csr_matrix, csc_matrix)):
        X_sp = np.asarray(X_sp.toarray(), dtype=np.float32)
    else:
        X_sp = np.asarray(X_sp, dtype=np.float32)

    # Row-normalise expression for cosine-like comparison
    row_norms = np.linalg.norm(X_sp, axis=1, keepdims=True)
    row_norms = np.clip(row_norms, EPS, None)
    X_normed = X_sp / row_norms

    # --- Build weighted adjacency ---
    adj = np.zeros((n, n), dtype=np.float32)
    alpha = float(np.clip(expression_weight, 0.0, 1.0))

    for i in range(n):
        for d, j in zip(dists[i, 1:], neighbors[i, 1:]):
            if not np.isfinite(d) or d <= 0:
                continue
            j = int(j)

            # Spatial weight: Gaussian kernel
            w_spatial = np.exp(-0.5 * (d / bandwidth) ** 2)

            # Expression weight: cosine similarity (clipped to [0, 1])
            w_expr = float(np.dot(X_normed[i], X_normed[j]))
            w_expr = max(w_expr, 0.0)  # negative similarity → 0

            # Combined weight
            w = (1.0 - alpha) * w_spatial + alpha * w_expr
            adj[i, j] = max(adj[i, j], w)
            adj[j, i] = max(adj[j, i], w)

    # Row-normalise
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    adj = adj / row_sum
    return adj


# =====================================================================
# Public API — Layer 2: Anisotropic differentiation gradient smoothing
# =====================================================================

def build_differentiation_gradient_inputs(
    adata_sc: sc.AnnData,
    adata_sp: sc.AnnData,
    spot_meta: pd.DataFrame,
    layer_order: Sequence[str],
    cluster_label: str,
    n_neighbors: int = 6,
    sigma: float = 2.0,
) -> np.ndarray:
    """Build differentiation-trajectory-aware gradient weights (Layer 2).

    v2.1 enhancement: instead of using region rank distance alone, this
    version derives a cell-type differentiation distance matrix from the
    scRNA-seq reference, then uses it to weight spatial neighbours.
    Neighbours whose region compositions differ *along* the expected
    differentiation trajectory are penalised less; neighbours that differ
    *against* the trajectory are penalised more.

    The differentiation distance between regions is estimated as the
    expression-space distance between their pseudobulk profiles — regions
    with similar transcriptomes (adjacent in differentiation) have small
    distance, and the penalty for composition differences between their
    spots is weaker.

    Parameters
    ----------
    adata_sc : AnnData
        Single-cell reference data.
    adata_sp : AnnData
        Spatial data.
    spot_meta : DataFrame
        Must contain columns 'x', 'y', 'layer_rank', 'layer_label'.
    layer_order : sequence of str
        Ordered region labels.
    cluster_label : str
        Cell type annotation column in adata_sc.
    n_neighbors : int
        Spatial neighbours.
    sigma : float
        Bandwidth of the Gaussian kernel for differentiation distance.
        Larger values → smoother gradient penalty.

    Returns
    -------
    gradient_weights : ndarray, shape (n_spots, n_spots)
        Weighted adjacency encoding differentiation continuity.
    """
    coords = spot_meta[["x", "y"]].to_numpy(dtype=np.float32)
    n = len(coords)
    n_regions = len(layer_order)

    # --- Compute region-level pseudobulk from scRNA-seq ---
    # Use spatial region pseudobulk as proxy for differentiation state
    shared_genes = list(
        set(adata_sc.var_names.str.lower()) & set(adata_sp.var_names.str.lower())
    )
    if len(shared_genes) < 10:
        # Fallback to rank-based if too few shared genes
        LOGGER.warning(
            "Too few shared genes (%d) for expression-based gradient; "
            "falling back to rank-based weights.", len(shared_genes),
        )
        return _build_rank_based_gradient(spot_meta, layer_order, n_neighbors)

    # Build region pseudobulk from spatial data
    sp_genes = [g for g in adata_sp.var_names if g.lower() in set(shared_genes)]
    X_sp = _to_dense_matrix(adata_sp, sp_genes)
    labels = spot_meta["layer_label"].astype(str).values
    region_profiles = _mean_by_group(X_sp, labels, layer_order)

    # Normalise profiles
    profile_norms = np.linalg.norm(region_profiles, axis=1, keepdims=True)
    profile_norms = np.clip(profile_norms, EPS, None)
    region_profiles_normed = region_profiles / profile_norms

    # Pairwise expression distance between regions
    # Using 1 - cosine_similarity as distance
    region_sim = region_profiles_normed @ region_profiles_normed.T
    region_dist = 1.0 - np.clip(region_sim, -1.0, 1.0)  # in [0, 2]

    # --- Map spots to regions ---
    region_to_idx = {str(r): i for i, r in enumerate(layer_order)}
    spot_region_idx = np.array([
        region_to_idx.get(str(l), 0) for l in spot_meta["layer_label"]
    ], dtype=int)

    # --- Build KNN graph ---
    tree = cKDTree(coords)
    k = min(n_neighbors + 1, n)
    dists, neighbors = tree.query(coords, k=k)
    if dists.ndim == 1:
        dists = dists[:, None]
        neighbors = neighbors[:, None]

    # --- Compute gradient weights ---
    weights = np.zeros((n, n), dtype=np.float32)
    sigma_sq = max(sigma ** 2, EPS)

    for i in range(n):
        ri = spot_region_idx[i]
        for d, j in zip(dists[i, 1:], neighbors[i, 1:]):
            if not np.isfinite(d) or d <= 0:
                continue
            j = int(j)
            rj = spot_region_idx[j]

            # Expression-based differentiation distance between regions
            diff_dist = float(region_dist[ri, rj])

            # Gaussian decay: same/similar regions → weight ≈ 1,
            # distant regions → weight ≈ 0
            w = np.exp(-diff_dist ** 2 / (2.0 * sigma_sq))
            weights[i, j] = max(weights[i, j], w)
            weights[j, i] = max(weights[j, i], w)

    # Row-normalise
    row_sum = weights.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    weights = weights / row_sum
    return weights


def _build_rank_based_gradient(
    spot_meta: pd.DataFrame,
    layer_order: Sequence[str],
    n_neighbors: int,
) -> np.ndarray:
    """Fallback: rank-based gradient weights (same as v2.0)."""
    coords = spot_meta[["x", "y"]].to_numpy(dtype=np.float32)
    ranks = spot_meta["layer_rank"].to_numpy(dtype=int)
    n = len(coords)
    n_regions = len(layer_order)
    denom = max(n_regions - 1, 1)

    tree = cKDTree(coords)
    k = min(n_neighbors + 1, n)
    dists, neighbors = tree.query(coords, k=k)
    if dists.ndim == 1:
        dists = dists[:, None]
        neighbors = neighbors[:, None]

    weights = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for d, j in zip(dists[i, 1:], neighbors[i, 1:]):
            if not np.isfinite(d) or d <= 0:
                continue
            j = int(j)
            rank_diff = abs(int(ranks[i]) - int(ranks[j])) / denom
            w = np.exp(-2.0 * rank_diff ** 2)
            weights[i, j] = max(weights[i, j], w)
            weights[j, i] = max(weights[j, i], w)

    row_sum = weights.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    weights = weights / row_sum
    return weights


# =====================================================================
# Public API — Layer 3: Adaptive per-spot regularisation strength
# =====================================================================

def build_adaptive_regularization_weights(
    adata_sp: sc.AnnData,
    spot_meta: pd.DataFrame,
    n_neighbors: int = 6,
    floor: float = 0.1,
    ceiling: float = 1.0,
) -> np.ndarray:
    """Compute per-spot adaptive regularisation strength (Layer 3).

    Homogeneous regions (low local expression heterogeneity) receive
    stronger smoothing (weight → ceiling), while heterogeneous boundary
    regions receive weaker smoothing (weight → floor).  This prevents
    the regularisation from washing out real biological transitions
    while still enforcing continuity in uniform tissue areas.

    Parameters
    ----------
    adata_sp : AnnData
        Spatial data.
    spot_meta : DataFrame
        Spot metadata.
    n_neighbors : int
        Number of spatial neighbours for heterogeneity estimation.
    floor : float
        Minimum regularisation weight (at boundaries).
    ceiling : float
        Maximum regularisation weight (in homogeneous areas).

    Returns
    -------
    adaptive_weights : ndarray, shape (n_spots,)
        Per-spot regularisation scaling factor in [floor, ceiling].
    """
    coords = extract_spatial_coordinates(adata_sp).loc[spot_meta.index]
    coord_arr = coords.to_numpy(dtype=np.float32)
    n = len(coord_arr)

    # Expression matrix
    X_sp = adata_sp[spot_meta.index].X
    if isinstance(X_sp, (csr_matrix, csc_matrix)):
        X_sp = np.asarray(X_sp.toarray(), dtype=np.float32)
    else:
        X_sp = np.asarray(X_sp, dtype=np.float32)

    # Row-normalise
    row_norms = np.linalg.norm(X_sp, axis=1, keepdims=True)
    row_norms = np.clip(row_norms, EPS, None)
    X_normed = X_sp / row_norms

    # KNN
    tree = cKDTree(coord_arr)
    k = min(n_neighbors + 1, n)
    dists, neighbors = tree.query(coord_arr, k=k)
    if dists.ndim == 1:
        dists = dists[:, None]
        neighbors = neighbors[:, None]

    # Compute local heterogeneity: mean (1 - cosine_similarity) to neighbours
    heterogeneity = np.zeros(n, dtype=np.float32)
    for i in range(n):
        sims = []
        for d, j in zip(dists[i, 1:], neighbors[i, 1:]):
            if not np.isfinite(d) or d <= 0:
                continue
            j = int(j)
            sim = float(np.dot(X_normed[i], X_normed[j]))
            sims.append(max(1.0 - sim, 0.0))
        if sims:
            heterogeneity[i] = float(np.mean(sims))

    # Normalise to [0, 1]
    h_min = heterogeneity.min()
    h_max = heterogeneity.max()
    if h_max - h_min > EPS:
        h_norm = (heterogeneity - h_min) / (h_max - h_min)
    else:
        h_norm = np.zeros_like(heterogeneity)

    # Invert: high heterogeneity → low weight, low heterogeneity → high weight
    adaptive_weights = ceiling - (ceiling - floor) * h_norm

    return adaptive_weights


# =====================================================================
# Public API — evaluation
# =====================================================================

def compute_layer_consistency_metrics(
    ct_pred: pd.DataFrame,
    spot_meta: pd.DataFrame,
    ct_layer_prior: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate how well predicted cell-type proportions respect region priors.

    Returns a DataFrame with out-of-band mass metrics per cell type.
    """
    ct_pred = ct_pred.copy()
    ct_pred = ct_pred.div(ct_pred.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    spot_meta = spot_meta.loc[ct_pred.index].copy()

    support = (ct_layer_prior.to_numpy(dtype=np.float32) > 0).astype(bool)
    layer_rank_spot = spot_meta["layer_rank"].to_numpy(dtype=int)
    rows = []
    for i, ct_name in enumerate(ct_pred.columns.astype(str)):
        values = ct_pred[ct_name].to_numpy(dtype=np.float32)
        support_layers = np.flatnonzero(support[i])
        if support_layers.size == 0:
            support_layers = np.array(
                [int(np.argmax(ct_layer_prior.to_numpy(dtype=np.float32)[i]))]
            )
        in_band = np.isin(layer_rank_spot, support_layers)
        out_mass = float(values[~in_band].sum())
        total = float(values.sum())
        rows.append({
            "cell_type": ct_name,
            "out_of_band_mass": out_mass,
            "out_of_band_fraction": out_mass / total if total > 0 else 0.0,
            "peak_layer": str(
                ct_layer_prior.columns[
                    int(np.argmax(ct_layer_prior.loc[ct_name].to_numpy(dtype=float)))
                ]
            ),
            "support_layers": ",".join(
                ct_layer_prior.columns[ct_layer_prior.loc[ct_name] > 0].astype(str)
            ),
        })
    return pd.DataFrame(rows).set_index("cell_type")
