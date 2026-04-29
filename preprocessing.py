"""
Preprocessing utilities for PlantDeconv.

Prepares single-cell and spatial AnnData objects for deconvolution by
finding shared genes, computing density priors, and building spatial
neighbourhood graphs.
"""

import logging

import numpy as np
import scanpy as sc

logging.getLogger().setLevel(logging.INFO)


def pp_adatas(adata_sc, adata_sp, genes=None, gene_to_lowercase=True):
    """
    Pre-process AnnDatas for deconvolution mapping.

    Steps performed:
    - Remove all-zero-valued genes.
    - Find the intersection of genes between adata_sc, adata_sp, and the
      optional marker gene list.
    - Compute density priors (uniform + RNA-count-based).
    - Compute spatial neighbourhood graph (requires squidpy).

    Args:
        adata_sc (AnnData): Single-cell data.
        adata_sp (AnnData): Spatial expression data.
        genes (list): Optional. List of genes to use. If None, all genes are used.
        gene_to_lowercase (bool): Optional. Convert gene names to lowercase. Default is True.

    Returns:
        None. Modifies adata_sc and adata_sp in-place by adding:
        - uns['training_genes'], uns['overlap_genes']
        - obs['rna_count_based_density'], obs['uniform_density'] (spatial only)
        - obsp['spatial_connectivities'], obsp['spatial_distances'] (spatial only)
    """
    # remove all-zero-valued genes
    sc.pp.filter_genes(adata_sc, min_cells=1)
    sc.pp.filter_genes(adata_sp, min_cells=1)

    if genes is None:
        genes = adata_sc.var.index

    # align gene names
    if gene_to_lowercase:
        adata_sc.var.index = [g.lower() for g in adata_sc.var.index]
        adata_sp.var.index = [g.lower() for g in adata_sp.var.index]
        genes = list(g.lower() for g in genes)

    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()

    # shared marker genes
    genes = list(set(genes) & set(adata_sc.var.index) & set(adata_sp.var.index))

    genes_train = genes
    adata_sc.uns["training_genes"] = genes_train
    adata_sp.uns["training_genes"] = genes_train
    logging.info(
        "{} training genes are saved in `uns``training_genes` of both AnnDatas.".format(
            len(genes_train)
        )
    )

    # overlap genes (sorted alphabetically for reproducible indexing)
    overlap_genes = np.sort(list(set(adata_sc.var.index) & set(adata_sp.var.index))).tolist()
    adata_sc.uns["overlap_genes"] = overlap_genes
    adata_sp.uns["overlap_genes"] = overlap_genes
    logging.info(
        "{} overlapped genes are saved in `uns``overlap_genes` of both AnnDatas.".format(
            len(overlap_genes)
        )
    )

    # density priors
    adata_sp.obs["uniform_density"] = np.ones(adata_sp.X.shape[0]) / adata_sp.X.shape[0]
    logging.info("Uniform density prior saved in `obs``uniform_density`.")

    rna_count_per_spot = np.array(adata_sp.X.sum(axis=1)).squeeze()
    adata_sp.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(rna_count_per_spot)
    logging.info("RNA-count density prior saved in `obs``rna_count_based_density`.")

    # spatial neighbourhood graph
    if "spatial" in adata_sp.obsm:
        logging.info("Computing spatial neighbourhood graph.")
        import squidpy as sq
        sq.gr.spatial_neighbors(adata_sp, set_diag=False)


def adata_to_cluster_expression(adata, cluster_label, scale=True, add_density=True):
    """
    Aggregate single-cell AnnData to cluster-level expression.

    Each cluster becomes one observation; expression is the mean (scale=False)
    or sum (scale=True) of cells in that cluster.

    Args:
        adata (AnnData): Single-cell data.
        cluster_label (str): Column in adata.obs for grouping.
        scale (bool): If True, use sum instead of mean. Default is True.
        add_density (bool): If True, add normalised cell counts as obs['cluster_density']. Default is True.

    Returns:
        AnnData: Cluster-level expression data.
    """
    import pandas as pd

    try:
        value_counts = adata.obs[cluster_label].value_counts(normalize=True)
    except KeyError:
        raise ValueError("Provided label must belong to adata.obs.")

    unique_labels = value_counts.index
    new_obs = pd.DataFrame({cluster_label: unique_labels})
    adata_ret = sc.AnnData(obs=new_obs, var=adata.var, uns=adata.uns)

    X_new = np.empty((len(unique_labels), adata.shape[1]))
    for index, label in enumerate(unique_labels):
        if not scale:
            X_new[index] = adata[adata.obs[cluster_label] == label].X.mean(axis=0)
        else:
            X_new[index] = adata[adata.obs[cluster_label] == label].X.sum(axis=0)
    adata_ret.X = X_new

    if add_density:
        adata_ret.obs["cluster_density"] = adata_ret.obs[cluster_label].map(
            lambda i: value_counts[i]
        )

    return adata_ret
