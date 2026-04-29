"""
Utility functions for PlantDeconv.

Includes helper functions for data manipulation, gene expression projection,
spatial comparison, cross-validation, and evaluation metrics.
"""

import logging
import warnings
from collections import defaultdict

import gzip
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import auc
from sklearn.model_selection import LeaveOneOut, KFold
from tqdm import tqdm

from . import mapping as mp
from . import preprocessing as pp

warnings.filterwarnings("ignore")
logger_ann = logging.getLogger("anndata")
logger_ann.disabled = True


def read_pickle(filename):
    """Read a (possibly gzipped) pickle file."""
    try:
        with gzip.open(filename, "rb") as f:
            return pickle.load(f)
    except OSError:
        with open(filename, "rb") as f:
            return pickle.load(f)


def annotate_gene_sparsity(adata):
    """
    Add gene sparsity annotation to adata.var['sparsity'].

    Sparsity is defined as 1 - (fraction of non-zero observations).
    """
    mask = adata.X != 0
    gene_sparsity = np.sum(mask, axis=0) / adata.n_obs
    gene_sparsity = np.asarray(gene_sparsity)
    gene_sparsity = 1 - np.reshape(gene_sparsity, (-1,))
    adata.var["sparsity"] = gene_sparsity


def get_matched_genes(prior_genes_names, sn_genes_names, excluded_genes=None):
    """
    Find shared genes between spatial and single-cell datasets.

    Returns:
        Tuple of (mask_prior_indices, mask_sn_indices, selected_genes).
    """
    prior_genes_names = np.array(prior_genes_names)
    sn_genes_names = np.array(sn_genes_names)

    mask_prior_indices = []
    mask_sn_indices = []
    selected_genes = []
    if excluded_genes is None:
        excluded_genes = []
    for index, i in enumerate(sn_genes_names):
        if i in excluded_genes:
            continue
        try:
            mask_prior_indices.append(np.argwhere(prior_genes_names == i)[0][0])
            mask_sn_indices.append(index)
            selected_genes.append(i)
        except IndexError:
            pass

    assert len(mask_prior_indices) == len(mask_sn_indices)
    return mask_prior_indices, mask_sn_indices, selected_genes


def one_hot_encoding(l, keep_aggregate=False):
    """
    One-hot encode a categorical sequence.

    Args:
        l (sequence): Categorical labels.
        keep_aggregate (bool): If True, keep the original column. Default is False.

    Returns:
        DataFrame with one-hot encoded columns.
    """
    df_enriched = pd.DataFrame({"cl": l})
    for i in l.unique():
        df_enriched[i] = list(map(int, df_enriched["cl"] == i))
    if not keep_aggregate:
        del df_enriched["cl"]
    return df_enriched


def project_cell_annotations(
    adata_map, adata_sp, annotation="cell_type", threshold=0.5
):
    """
    Transfer cell annotations from single-cell onto spatial data.

    Args:
        adata_map (AnnData): Mapping result.
        adata_sp (AnnData): Spatial data.
        annotation (str): Column in adata_map.obs. Default is 'cell_type'.
        threshold (float): Filter threshold for constrained mode. Default is 0.5.

    Returns:
        None. Updates adata_sp.obsm['plantdeconv_ct_pred'].
    """
    df = one_hot_encoding(adata_map.obs[annotation])
    if "F_out" in adata_map.obs.keys():
        df_ct_prob = adata_map[adata_map.obs["F_out"] > threshold]

    df_ct_prob = adata_map.X.T @ df
    df_ct_prob.index = adata_map.var.index

    adata_sp.obsm["plantdeconv_ct_pred"] = df_ct_prob
    logging.info(
        "Cell type predictions saved in `obsm``plantdeconv_ct_pred` of the spatial AnnData."
    )


def create_segment_cell_df(adata_sp):
    """
    Create segmentation cell DataFrame from spatial data.

    Requires adata_sp.obsm['image_features'] from squidpy.
    """
    if "image_features" not in adata_sp.obsm.keys():
        raise ValueError(
            "Missing image features. Run `squidpy.im.calculate_image_features`."
        )

    centroids = adata_sp.obsm["image_features"][["segmentation_centroid"]].copy()
    centroids["centroids_idx"] = [
        np.array([f"{k}_{j}" for j in np.arange(i)], dtype="object")
        for k, i in zip(
            adata_sp.obs.index.values,
            adata_sp.obsm["image_features"]["segmentation_label"],
        )
    ]
    centroids_idx = centroids.explode("centroids_idx")
    centroids_coords = centroids.explode("segmentation_centroid")
    segmentation_df = pd.DataFrame(
        centroids_coords["segmentation_centroid"].to_list(),
        columns=["y", "x"],
        index=centroids_coords.index,
    )
    segmentation_df["centroids"] = centroids_idx["centroids_idx"].values
    segmentation_df.index.set_names("spot_idx", inplace=True)
    segmentation_df.reset_index(drop=False, inplace=True)

    adata_sp.uns["plantdeconv_cell_segmentation"] = segmentation_df
    adata_sp.obsm["plantdeconv_spot_centroids"] = centroids["centroids_idx"]
    logging.info("Cell segmentation saved in `uns``plantdeconv_cell_segmentation`.")


def count_cell_annotations(
    adata_map, adata_sc, adata_sp, annotation="cell_type", threshold=0.5,
):
    """
    Count cells per annotation in each spatial voxel.

    Updates adata_sp.obsm['plantdeconv_ct_count'].
    """
    if "spatial" not in adata_sp.obsm.keys():
        raise ValueError("Missing spatial coordinates in adata_sp.obsm['spatial'].")

    if "image_features" not in adata_sp.obsm.keys():
        raise ValueError("Missing image features. Run squidpy image feature extraction.")

    if (
        "plantdeconv_cell_segmentation" not in adata_sp.uns.keys()
        or "plantdeconv_spot_centroids" not in adata_sp.obsm.keys()
    ):
        raise ValueError("Missing segmentation data. Run `create_segment_cell_df`.")

    xs = adata_sp.obsm["spatial"][:, 1]
    ys = adata_sp.obsm["spatial"][:, 0]
    cell_count = adata_sp.obsm["image_features"]["segmentation_label"]

    centroids = adata_sp.obsm["plantdeconv_spot_centroids"]

    df_vox_cells = pd.DataFrame(
        data={"x": xs, "y": ys, "cell_n": cell_count, "centroids": centroids},
        index=list(adata_sp.obs.index),
    )

    resulting_voxels = np.argmax(adata_map.X, axis=1)

    if "F_out" in adata_map.obs.keys():
        filtered_voxels_to_types = [
            (j, adata_sc.obs[annotation][k])
            for i, j, k in zip(
                adata_map.obs["F_out"], resulting_voxels, range(len(adata_sc))
            )
            if i > threshold
        ]
        vox_ct = filtered_voxels_to_types
    else:
        vox_ct = list(zip(resulting_voxels, adata_sc.obs[annotation]))

    df_classes = one_hot_encoding(adata_sc.obs[annotation])
    for index, i in enumerate(df_classes.columns):
        df_vox_cells[i] = 0

    for k, v in vox_ct:
        df_vox_cells.iloc[k, df_vox_cells.columns.get_loc(v)] += 1

    adata_sp.obsm["plantdeconv_ct_count"] = df_vox_cells
    logging.info("Cell type counts saved in `obsm``plantdeconv_ct_count`.")


def deconvolve_cell_annotations(adata_sp, filter_cell_annotation=None):
    """
    Assign cell annotations to segmented cells.

    Returns an AnnData where each observation is a segmented cell.
    """
    if (
        "plantdeconv_ct_count" not in adata_sp.obsm.keys()
        or "plantdeconv_cell_segmentation" not in adata_sp.uns.keys()
    ):
        raise ValueError("Missing data. Run `count_cell_annotations`.")

    segmentation_df = adata_sp.uns["plantdeconv_cell_segmentation"]

    if filter_cell_annotation is None:
        filter_cell_annotation = pd.unique(
            list(adata_sp.obsm["plantdeconv_ct_pred"].columns)
        )
    else:
        filter_cell_annotation = pd.unique(filter_cell_annotation)

    df_vox_cells = adata_sp.obsm["plantdeconv_ct_count"]
    cell_types_mapped = df_to_cell_types(df_vox_cells, filter_cell_annotation)
    df_list = []
    for k in cell_types_mapped.keys():
        df = pd.DataFrame({"centroids": np.array(cell_types_mapped[k], dtype="object")})
        df["cluster"] = k
        df_list.append(df)
    cluster_df = pd.concat(df_list, axis=0)
    cluster_df.reset_index(inplace=True, drop=True)

    merged_df = segmentation_df.merge(cluster_df, on="centroids", how="inner")
    merged_df.drop(columns="spot_idx", inplace=True)
    merged_df.drop_duplicates(inplace=True)
    merged_df.dropna(inplace=True)
    merged_df.reset_index(inplace=True, drop=True)

    adata_segment = sc.AnnData(np.zeros(merged_df.shape), obs=merged_df)
    adata_segment.obsm["spatial"] = merged_df[["y", "x"]].to_numpy()
    adata_segment.uns = adata_sp.uns

    return adata_segment


def project_genes(adata_map, adata_sc, cluster_label=None, scale=True):
    """
    Project gene expression from single-cell onto spatial coordinates.

    Returns:
        AnnData: Spot-by-gene predicted expression data.
    """
    adata_sc.var.index = [g.lower() for g in adata_sc.var.index]
    adata_sc.var_names_make_unique()
    sc.pp.filter_genes(adata_sc, min_cells=1)

    if cluster_label:
        adata_sc = pp.adata_to_cluster_expression(adata_sc, cluster_label, scale=scale)

    if not adata_map.obs.index.equals(adata_sc.obs.index):
        raise ValueError("The two AnnDatas need to have the same obs index.")
    if hasattr(adata_sc.X, "toarray"):
        adata_sc.X = adata_sc.X.toarray()
    X_space = adata_map.X.T @ adata_sc.X
    adata_ge = sc.AnnData(
        X=X_space, obs=adata_map.var, var=adata_sc.var, uns=adata_sc.uns
    )
    training_genes = adata_map.uns["train_genes_df"].index.values
    adata_ge.var["is_training"] = adata_ge.var.index.isin(training_genes)
    return adata_ge


def compare_spatial_geneexp(adata_ge, adata_sp, adata_sc=None, genes=None):
    """
    Compare predicted spatial gene expression with actual spatial data.

    Returns:
        DataFrame with cosine similarity scores per gene.
    """
    logger_root = logging.getLogger()
    logger_root.disabled = True

    if not {"training_genes", "overlap_genes"}.issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing parameters. Run `pp_adatas()`.")
    if not {"training_genes", "overlap_genes"}.issubset(set(adata_ge.uns.keys())):
        raise ValueError("Missing parameters. Use `project_genes()` to get adata_ge.")

    assert list(adata_sp.uns["overlap_genes"]) == list(adata_ge.uns["overlap_genes"])

    if genes is None:
        overlap_genes = adata_ge.uns["overlap_genes"]
    else:
        overlap_genes = genes

    annotate_gene_sparsity(adata_sp)

    cos_sims = []
    if hasattr(adata_ge.X, "toarray"):
        X_1 = adata_ge[:, overlap_genes].X.toarray()
    else:
        X_1 = adata_ge[:, overlap_genes].X

    if hasattr(adata_sp.X, "toarray"):
        X_2 = adata_sp[:, overlap_genes].X.toarray()
    else:
        X_2 = adata_sp[:, overlap_genes].X

    for v1, v2 in zip(X_1.T, X_2.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    df_g = pd.DataFrame(cos_sims, overlap_genes, columns=["score"])
    for adata in [adata_ge, adata_sp]:
        if "is_training" in adata.var.keys():
            df_g["is_training"] = adata.var.is_training

    df_g["sparsity_sp"] = adata_sp[:, overlap_genes].var.sparsity

    if adata_sc is not None:
        if not {"training_genes", "overlap_genes"}.issubset(set(adata_sc.uns.keys())):
            raise ValueError("Missing parameters. Run `pp_adatas()`.")
        assert list(adata_sc.uns["overlap_genes"]) == list(adata_sp.uns["overlap_genes"])
        annotate_gene_sparsity(adata_sc)
        df_g = df_g.merge(
            pd.DataFrame(adata_sc[:, overlap_genes].var["sparsity"]),
            left_index=True, right_index=True,
        )
        df_g.rename({"sparsity": "sparsity_sc"}, inplace=True, axis="columns")
        df_g["sparsity_diff"] = df_g["sparsity_sp"] - df_g["sparsity_sc"]

    if genes is not None:
        df_g = df_g.loc[genes]

    df_g = df_g.sort_values(by="score", ascending=False)
    return df_g


def df_to_cell_types(df, cell_types):
    """Assign cell coordinates to cell types for deconvolution."""
    df_cum_sums = df[cell_types].cumsum(axis=1)
    df_c = df.copy()
    for i in df_cum_sums.columns:
        df_c[i] = df_cum_sums[i]

    cell_types_mapped = defaultdict(list)
    for i_index, i in enumerate(cell_types):
        for j_index, j in df_c.iterrows():
            start_ind = 0 if i_index == 0 else j[cell_types[i_index - 1]]
            end_ind = j[i]
            cell_types_mapped[i].extend(j["centroids"][start_ind:end_ind].tolist())
    return cell_types_mapped


def cell_type_mapping(adata_map, cell_types_key="cell_types"):
    """
    Compute cell type mapping from the cell mapping matrix.

    Updates adata_map.varm['ct_map'].
    """
    df = one_hot_encoding(adata_map.obs[cell_types_key])
    if "F_out" in adata_map.obs.keys():
        df_ct_prob = adata_map[adata_map.obs["F_out"] >= 0.5].X.T @ df
    else:
        df_ct_prob = adata_map.X.T @ df
    df_ct_prob.index = adata_map.var.index
    vmin = df_ct_prob.min()
    vmax = df_ct_prob.max()
    df_ct_prob = (df_ct_prob - vmin) / (vmax - vmin)
    adata_map.varm["ct_map"] = df_ct_prob


def cv_data_gen(adata_sc, adata_sp, cv_mode="loo"):
    """Generate train/test gene splits for cross-validation."""
    if "training_genes" not in adata_sc.uns.keys():
        raise ValueError("Missing parameters. Run `pp_adatas()`.")
    if "training_genes" not in adata_sp.uns.keys():
        raise ValueError("Missing parameters. Run `pp_adatas()`.")
    if not list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"]):
        raise ValueError("Unmatched training_genes. Run `pp_adatas()`.")

    genes_array = np.array(adata_sp.uns["training_genes"])
    if cv_mode == "loo":
        cv = LeaveOneOut()
    elif cv_mode == "10fold":
        cv = KFold(n_splits=10)

    for train_idx, test_idx in cv.split(genes_array):
        train_genes = list(genes_array[train_idx])
        test_genes = list(genes_array[test_idx])
        yield train_genes, test_genes


def cross_val(
    adata_sc, adata_sp,
    cluster_label=None, mode="clusters", scale=True,
    lambda_d=0, lambda_g1=1, lambda_g2=0, lambda_r=0,
    lambda_count=1, lambda_f_reg=1, target_count=None,
    num_epochs=1000, device="cuda:0", learning_rate=0.1,
    cv_mode="loo", return_gene_pred=False,
    density_prior=None, random_state=None, verbose=False,
):
    """
    Run cross-validation for mapping quality assessment.

    Returns:
        dict with avg_test_score and avg_train_score.
        Optionally also returns predicted expression AnnData and test gene DataFrame.
    """
    logger_root = logging.getLogger()
    logger_root.disabled = True

    test_genes_list = []
    test_pred_list = []
    test_score_list = []
    train_score_list = []
    test_df_list = []
    curr_cv_set = 1

    if cv_mode == "loo":
        length = len(list(adata_sc.uns["training_genes"]))
    elif cv_mode == "10fold":
        length = 10

    if mode == "clusters":
        adata_sc_agg = pp.adata_to_cluster_expression(adata_sc, cluster_label, scale)

    for train_genes, test_genes in tqdm(
        cv_data_gen(adata_sc, adata_sp, cv_mode), total=length
    ):
        adata_map = mp.map_cells_to_space(
            adata_sc=adata_sc, adata_sp=adata_sp,
            cv_train_genes=train_genes, mode=mode, device=device,
            learning_rate=learning_rate, num_epochs=num_epochs,
            cluster_label=cluster_label, scale=scale,
            lambda_d=lambda_d, lambda_g1=lambda_g1, lambda_g2=lambda_g2,
            lambda_r=lambda_r, lambda_count=lambda_count,
            lambda_f_reg=lambda_f_reg, target_count=target_count,
            random_state=random_state, verbose=False, density_prior=density_prior,
        )

        cv_genes = train_genes + test_genes
        adata_ge = project_genes(
            adata_map, adata_sc[:, cv_genes], cluster_label=cluster_label, scale=scale,
        )

        if cv_mode == "loo" and return_gene_pred:
            adata_ge_test = adata_ge[:, test_genes].X.T
            test_pred_list.append(adata_ge_test)

        if mode == "clusters":
            df_g = compare_spatial_geneexp(adata_ge, adata_sp, adata_sc_agg, cv_genes)
        else:
            df_g = compare_spatial_geneexp(adata_ge, adata_sp, adata_sc, cv_genes)

        test_df = df_g[df_g.index.isin(test_genes)]
        test_score = df_g.loc[test_genes]["score"].mean()
        train_score = float(list(adata_map.uns["training_history"]["main_loss"])[-1])

        test_genes_list.append(test_genes)
        test_score_list.append(test_score)
        train_score_list.append(train_score)
        test_df_list.append(test_df)

        if verbose:
            msg = "cv set: {}----train score: {:.3f}----test score: {:.3f}".format(
                curr_cv_set, train_score, test_score
            )
            print(msg)

        curr_cv_set += 1

    avg_test_score = np.nanmean(test_score_list)
    avg_train_score = np.nanmean(train_score_list)

    cv_dict = {
        "avg_test_score": avg_test_score,
        "avg_train_score": avg_train_score,
    }

    print("cv avg test score {:.3f}".format(avg_test_score))
    print("cv avg train score {:.3f}".format(avg_train_score))

    if cv_mode == "loo" and return_gene_pred:
        test_gene_df = pd.concat(test_df_list, axis=0)
        adata_ge_cv = sc.AnnData(
            X=np.squeeze(test_pred_list).T,
            obs=adata_sp.obs.copy(),
            var=pd.DataFrame(
                test_score_list,
                columns=["test_score"],
                index=np.squeeze(test_genes_list),
            ),
        )
        return cv_dict, adata_ge_cv, test_gene_df

    return cv_dict


def eval_metric(df_all_genes, test_genes=None):
    """
    Compute evaluation metrics (avg scores and AUC).

    Returns:
        Tuple of (metric_dict, auc_coordinates).
    """
    if test_genes is not None:
        if not set(test_genes).issubset(set(df_all_genes.index.values)):
            raise ValueError("test_genes should be subset of genes in input dataframe.")
        test_genes = np.unique(test_genes)
    else:
        test_genes = list(
            set(df_all_genes[df_all_genes["is_training"] == False].index.values)
        )

    test_gene_scores = df_all_genes.loc[test_genes]["score"]
    test_gene_sparsity_sp = df_all_genes.loc[test_genes]["sparsity_sp"]
    test_score_avg = test_gene_scores.mean()
    train_score_avg = df_all_genes[df_all_genes["is_training"] == True]["score"].mean()

    test_score_sps_sp_g2 = np.sum(
        (test_gene_scores * (1 - test_gene_sparsity_sp))
        / (1 - test_gene_sparsity_sp).sum()
    )

    xs = list(test_gene_scores)
    ys = list(test_gene_sparsity_sp)
    pol_deg = 2
    pol_cs = np.polyfit(xs, ys, pol_deg)
    pol_xs = np.linspace(0, 1, 10)
    pol = np.poly1d(pol_cs)
    pol_ys = [pol(x) for x in pol_xs]

    if pol_ys[0] > 1:
        pol_ys[0] = 1

    roots = pol.r
    root = None
    for i in range(len(roots)):
        if np.isreal(roots[i]) and roots[i] <= 1 and roots[i] >= 0:
            root = roots[i]
            break

    if root is not None:
        pol_xs = np.append(pol_xs, root)
        pol_ys = np.append(pol_ys, 0)

    np.append(pol_xs, 1)
    np.append(pol_ys, pol(1))

    del_idx = []
    for i in range(len(pol_xs)):
        if pol_xs[i] < 0 or pol_ys[i] < 0 or pol_xs[i] > 1 or pol_ys[i] > 1:
            del_idx.append(i)

    pol_xs = [x for x in pol_xs if list(pol_xs).index(x) not in del_idx]
    pol_ys = [y for y in pol_ys if list(pol_ys).index(y) not in del_idx]

    auc_test_score = np.real(auc(pol_xs, pol_ys))

    metric_dict = {
        "avg_test_score": test_score_avg,
        "avg_train_score": train_score_avg,
        "sp_sparsity_score": test_score_sps_sp_g2,
        "auc_score": auc_test_score,
    }

    auc_coordinates = ((pol_xs, pol_ys), (xs, ys))
    return metric_dict, auc_coordinates


# DEPRECATED
def transfer_annotations_prob(mapping_matrix, to_transfer):
    """Transfer cell annotations onto space through a mapping matrix."""
    return mapping_matrix.transpose() @ to_transfer


def transfer_annotations_prob_filter(mapping_matrix, filter, to_transfer):
    """Transfer annotations with filter."""
    tt = to_transfer * filter[:, np.newaxis]
    return mapping_matrix.transpose() @ tt
