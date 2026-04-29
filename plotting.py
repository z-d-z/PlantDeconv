"""
Plotting utilities for PlantDeconv.

Provides diagnostic plots for training results, spatial visualisation of
cell-type predictions, and pie-chart maps for deconvolution results.
"""

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.patches import Wedge
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import entropy

from . import utils as ut
from . import preprocessing as pp

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def q_value(data, perc):
    """
    Compute min and max values according to percentile for colormaps.

    Args:
        data (array): Input values.
        perc (float): Percentile between 0 and 100.

    Returns:
        Tuple of (vmin, vmax).
    """
    vmin = np.nanpercentile(data, perc)
    vmax = np.nanpercentile(data, 100 - perc)
    return vmin, vmax


def ordered_predictions(xs, ys, preds, reverse=False):
    """
    Order 2D points by associated prediction values.

    Args:
        xs, ys (sequence): Coordinates.
        preds (sequence): Prediction values.
        reverse (bool): Sort descending if True. Default is False.

    Returns:
        Tuple of sorted (xs, ys, preds).
    """
    assert len(xs) == len(ys) == len(preds)
    return list(
        zip(
            *[
                (x, y, z)
                for x, y, z in sorted(
                    zip(xs, ys, preds), key=lambda pair: pair[2], reverse=reverse
                )
            ]
        )
    )


def convert_adata_array(adata):
    """Convert sparse AnnData.X to dense array in-place."""
    if isinstance(adata.X, (csc_matrix, csr_matrix)):
        adata.X = adata.X.toarray()


def construct_obs_plot(df_plot, adata, perc=0, suffix=None):
    """Clip, normalise, and attach prediction columns to adata.obs."""
    df_plot = df_plot.clip(df_plot.quantile(perc), df_plot.quantile(1 - perc), axis=1)
    df_plot = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())
    if suffix:
        df_plot = df_plot.add_suffix(" ({})".format(suffix))
    adata.obs = pd.concat([adata.obs, df_plot], axis=1)


# ---------------------------------------------------------------------------
# Training diagnostic plots
# ---------------------------------------------------------------------------

def plot_training_scores(adata_map, bins=10, alpha=0.7):
    """
    Plot the 4-panel training diagnosis plot.

    Shows training score distribution and score vs. sparsity scatter plots.

    Args:
        adata_map (AnnData): Mapping result with uns['train_genes_df'].
        bins (int): Histogram bins. Default is 10.
        alpha (float): Scatter opacity. Default is 0.7.
    """
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    df = adata_map.uns["train_genes_df"]
    axs_f = axs.flatten()

    axs_f[0].set_ylim([0.0, 1.0])
    for i in range(1, len(axs_f)):
        axs_f[i].set_xlim([0.0, 1.0])
        axs_f[i].set_ylim([0.0, 1.0])

    sns.histplot(data=df, y="train_score", bins=bins, ax=axs_f[0], color="coral")

    axs_f[1].set_title("score vs sparsity (single cells)")
    sns.scatterplot(data=df, y="train_score", x="sparsity_sc", ax=axs_f[1], alpha=alpha, color="coral")

    axs_f[2].set_title("score vs sparsity (spatial)")
    sns.scatterplot(data=df, y="train_score", x="sparsity_sp", ax=axs_f[2], alpha=alpha, color="coral")

    axs_f[3].set_title("score vs sparsity (sp - sc)")
    sns.scatterplot(data=df, y="train_score", x="sparsity_diff", ax=axs_f[3], alpha=alpha, color="coral")

    plt.tight_layout()


def plot_gene_sparsity(
    adata_1, adata_2, xlabel="adata_1", ylabel="adata_2", genes=None, s=1
):
    """
    Compare gene sparsity between two AnnData objects.

    Args:
        adata_1, adata_2 (AnnData): Input datasets.
        xlabel, ylabel (str): Axis labels.
        genes (list): Optional gene subset.
        s (float): Marker size. Default is 1.
    """
    logging.info("Pre-processing AnnDatas...")
    pp.pp_adatas(adata_1, adata_2, genes=genes)
    assert adata_1.uns["training_genes"] == adata_2.uns["training_genes"]
    training_genes = adata_1.uns["training_genes"]

    logging.info("Annotating sparsity...")
    ut.annotate_gene_sparsity(adata_1)
    ut.annotate_gene_sparsity(adata_2)
    xs = adata_1[:, training_genes].var["sparsity"].values
    ys = adata_2[:, training_genes].var["sparsity"].values
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect(1)
    ax.set_xlabel("sparsity (" + xlabel + ")")
    ax.set_ylabel("sparsity (" + ylabel + ")")
    ax.set_title(f"Gene sparsity ({len(xs)} genes)")
    ax.scatter(xs, ys, s=s, marker="x")


def plot_test_scores(df_gene_score, bins=10, alpha=0.7):
    """
    Plot gene-level test scores with sparsity.

    Args:
        df_gene_score (DataFrame): From compare_spatial_geneexp with adata_sc.
        bins (int): Histogram bins. Default is 10.
        alpha (float): Scatter opacity. Default is 0.7.
    """
    if not {"score", "sparsity_sc", "sparsity_sp", "sparsity_diff"}.issubset(
        set(df_gene_score.columns)
    ):
        raise ValueError(
            "Missing columns. Run `compare_spatial_geneexp` with adata_sc to produce complete input."
        )

    if "is_training" in df_gene_score.keys():
        df = df_gene_score[df_gene_score["is_training"] == False]
    else:
        df = df_gene_score

    df = df.copy()
    df.rename({"score": "test_score"}, axis="columns", inplace=True)

    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    axs_f = axs.flatten()

    axs_f[0].set_ylim([0.0, 1.0])
    for i in range(1, len(axs_f)):
        axs_f[i].set_xlim([0.0, 1.0])
        axs_f[i].set_ylim([0.0, 1.0])

    sns.histplot(data=df, y="test_score", bins=bins, ax=axs_f[0])
    axs_f[1].set_title("score vs sparsity (single cells)")
    sns.scatterplot(data=df, y="test_score", x="sparsity_sc", ax=axs_f[1], alpha=alpha)
    axs_f[2].set_title("score vs sparsity (spatial)")
    sns.scatterplot(data=df, y="test_score", x="sparsity_sp", ax=axs_f[2], alpha=alpha)
    axs_f[3].set_title("score vs sparsity (sp - sc)")
    sns.scatterplot(data=df, y="test_score", x="sparsity_diff", ax=axs_f[3], alpha=alpha)
    plt.tight_layout()


def plot_auc(df_all_genes, test_genes=None):
    """
    Plot AUC curve for model evaluation.

    Args:
        df_all_genes (DataFrame): From compare_spatial_geneexp.
        test_genes (list): Optional test gene subset.
    """
    metric_dict, ((pol_xs, pol_ys), (xs, ys)) = ut.eval_metric(df_all_genes, test_genes)

    fig = plt.figure()
    plt.figure(figsize=(6, 5))
    plt.plot(pol_xs, pol_ys, c="r")
    sns.scatterplot(x=xs, y=ys, alpha=0.5, edgecolors="face")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().set_aspect(0.5)
    plt.xlabel("score")
    plt.ylabel("spatial sparsity")
    plt.tick_params(axis="both", labelsize=8)
    plt.title("Prediction on test transcriptome")

    textstr = "auc_score={}".format(np.round(metric_dict["auc_score"], 3))
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.3)
    plt.text(0.03, 0.1, textstr, fontsize=11, verticalalignment="top", bbox=props)


# ---------------------------------------------------------------------------
# Spatial cell-type annotation plots
# ---------------------------------------------------------------------------

def plot_cell_annotation_sc(
    adata_sp,
    annotation_list,
    x="x",
    y="y",
    spot_size=None,
    scale_factor=None,
    perc=0,
    alpha_img=1.0,
    bw=False,
    ax=None,
):
    """
    Plot spatial cell-type prediction heatmaps (Visium-compatible).

    Args:
        adata_sp (AnnData): Spatial data with obsm['plantdeconv_ct_pred'].
        annotation_list (list): Cell types to plot.
        x, y (str): Coordinate column names.
        spot_size, scale_factor: Visium plot parameters.
        perc (float): Clipping percentile.
        alpha_img (float): Background image opacity.
        bw (bool): Greyscale background.
        ax: Matplotlib axes.
    """
    adata_sp.obs.drop(annotation_list, inplace=True, errors="ignore", axis=1)

    # Support both old and new key names
    pred_key = "plantdeconv_ct_pred"
    if pred_key not in adata_sp.obsm:
        pred_key = "tangram_ct_pred"
    df = adata_sp.obsm[pred_key][annotation_list]
    construct_obs_plot(df, adata_sp, perc=perc)

    if "spatial" not in adata_sp.obsm.keys():
        coords = [[x_val, y_val] for x_val, y_val in zip(adata_sp.obs[x].values, adata_sp.obs[y].values)]
        adata_sp.obsm["spatial"] = np.array(coords)

    if "spatial" not in adata_sp.uns.keys() and spot_size is None and scale_factor is None:
        raise ValueError("Spot Size and Scale Factor cannot be None when uns['spatial'] does not exist.")

    if "spatial" in adata_sp.uns.keys() and spot_size is not None and scale_factor is not None:
        raise ValueError("Spot Size and Scale Factor should be None when uns['spatial'] exists.")

    sc.pl.spatial(
        adata_sp, color=annotation_list, cmap="viridis", show=False, frameon=False,
        spot_size=spot_size, scale_factor=scale_factor, alpha_img=alpha_img, bw=bw, ax=ax,
    )

    adata_sp.obs.drop(annotation_list, inplace=True, errors="ignore", axis=1)


def plot_cell_annotation(
    adata_map,
    adata_sp,
    annotation="cell_type",
    x="x",
    y="y",
    nrows=1,
    ncols=1,
    s=5,
    cmap="viridis",
    subtitle_add=False,
    robust=False,
    perc=0,
    invert_y=True,
):
    """
    Visualise spatial cell-type probability maps.

    Args:
        adata_map (AnnData): Mapping result.
        adata_sp (AnnData): Spatial data.
        annotation (str): Column in adata_map.obs. Default is 'cell_type'.
        x, y (str): Coordinate columns. Default is 'x', 'y'.
        nrows, ncols (int): Subplot layout.
        s (float): Marker size. Default is 5.
        cmap (str): Colourmap. Default is 'viridis'.
        subtitle_add (bool): Add score subtitle. Default is False.
        robust (bool): Use percentile clipping. Default is False.
        perc (float): Clipping percentile.
        invert_y (bool): Invert y-axis. Default is True.
    """
    df = ut.one_hot_encoding(adata_map.obs[annotation])
    if "F_out" in adata_map.obs.keys():
        df_ct_prob = adata_map[adata_map.obs["F_out"] > 0.5].X.T @ df
    else:
        df_ct_prob = adata_map.X.T @ df
    df_ct_prob.index = adata_map.var.index

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows * ncols == 1:
        axs = np.array([axs])
    axs_f = axs.flatten()

    ii = 0
    for ann in df_ct_prob.columns:
        xs, ys, vs = ordered_predictions(
            adata_sp.obs[x], adata_sp.obs[y], np.array(df_ct_prob[ann]),
        )
        if robust:
            vmin, vmax = q_value(vs, perc=perc)
        else:
            vmin, vmax = q_value(vs, perc=0)
        axs_f[ii].scatter(xs, ys, c=vs, cmap=cmap, s=s, vmin=vmin, vmax=vmax)

        title_str = str(ann)
        if subtitle_add:
            title_str += " (train_score: {:.2f})".format(
                adata_map.uns["train_genes_df"]["train_score"].get(ann, float("nan"))
            )
        axs_f[ii].set_title(title_str, fontsize=10)
        axs_f[ii].axis("off")
        axs_f[ii].set_aspect(1)

        if invert_y:
            axs_f[ii].invert_yaxis()
        ii += 1

    plt.tight_layout()


def plot_genes(
    genes,
    adata_measured,
    adata_predicted,
    x="x",
    y="y",
    s=5,
    log=False,
    cmap="viridis",
    robust=False,
    perc=0,
    invert_y=True,
):
    """
    Compare measured vs predicted gene expression spatially.

    Args:
        genes (list): Gene names to plot.
        adata_measured (AnnData): Actual spatial data.
        adata_predicted (AnnData): Predicted spatial data.
        x, y (str): Coordinate columns.
        s (float): Marker size.
        log (bool): Apply log transform.
        cmap (str): Colourmap.
        robust (bool): Percentile clipping.
        perc (float): Clipping percentile.
        invert_y (bool): Invert y-axis.
    """
    convert_adata_array(adata_measured)

    adata_measured.var.index = [g.lower() for g in adata_measured.var.index]
    adata_predicted.var.index = [g.lower() for g in adata_predicted.var.index]

    fig, ax = plt.subplots(figsize=(4, 0.4))
    fig.subplots_adjust(top=0.5)
    cmap_obj = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    mpl.colorbar.ColorbarBase(ax, cmap=cmap_obj, norm=norm, orientation="horizontal", label="Expression Level")

    fig, axs = plt.subplots(nrows=len(genes), ncols=2, figsize=(6, len(genes) * 3))

    for ix, gene in enumerate(genes):
        if gene not in adata_measured.var.index:
            vs = np.zeros_like(np.array(adata_measured[:, 0].X).flatten())
        else:
            vs = np.array(adata_measured[:, gene].X).flatten()

        xs, ys, vs = ordered_predictions(adata_measured.obs[x], adata_measured.obs[y], vs)
        if log:
            vs = np.log(1 + np.asarray(vs))
        axs[ix, 0].scatter(xs, ys, c=vs, cmap=cmap, s=s)
        axs[ix, 0].set_title(gene + " (measured)")
        axs[ix, 0].axis("off")
        axs[ix, 0].set_aspect(1)

        xs, ys, vs = ordered_predictions(
            adata_predicted.obs[x], adata_predicted.obs[y],
            np.array(adata_predicted[:, gene].X).flatten(),
        )
        if robust:
            vmin, vmax = q_value(vs, perc=perc)
        else:
            vmin, vmax = q_value(vs, perc=0)
        if log:
            vs = np.log(1 + np.asarray(vs))
        axs[ix, 1].scatter(xs, ys, c=vs, cmap=cmap, s=s, vmin=vmin, vmax=vmax)
        axs[ix, 1].set_title(gene + " (predicted)")
        axs[ix, 1].axis("off")
        axs[ix, 1].set_aspect(1)

        if invert_y:
            axs[ix, 0].invert_yaxis()
            axs[ix, 1].invert_yaxis()


def quick_plot_gene(
    gene, adata, x="x", y="y", s=50, log=False, cmap="viridis", robust=False, perc=0
):
    """Quickly plot a single gene's spatial expression."""
    if not robust and perc != 0:
        raise ValueError("perc must be zero when robust is False.")
    if robust and perc == 0:
        raise ValueError("perc cannot be zero when robust is True.")

    xs, ys, vs = ordered_predictions(
        adata.obs[x], adata.obs[y], np.array(adata[:, gene].X).flatten()
    )
    if robust:
        vmin, vmax = q_value(vs, perc=perc)
    else:
        vmin, vmax = q_value(vs, perc=0)
    if log:
        vs = np.log(1 + np.asarray(vs))
    plt.scatter(xs, ys, c=vs, cmap=cmap, s=s, vmin=vmin, vmax=vmax)


def plot_annotation_entropy(adata_map, annotation="cell_type"):
    """
    Plot entropy box plot by cell annotation.

    Args:
        adata_map (AnnData): Mapping result.
        annotation (str): Column in adata_map.obs. Default is 'cell_type'.
    """
    adata_map.obs["entropy"] = entropy(adata_map.X, base=adata_map.X.shape[1], axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.set_ylim(0, 1)
    sns.boxenplot(x=annotation, y="entropy", data=adata_map.obs, ax=ax)
    plt.xticks(rotation=30)


# ---------------------------------------------------------------------------
# Pie chart map for deconvolution results
# ---------------------------------------------------------------------------

def _normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    row_sum = df.sum(axis=1).replace(0, np.nan)
    return df.div(row_sum, axis=0).fillna(0.0)


def _infer_predict_df(adata: sc.AnnData) -> pd.DataFrame:
    for key in ["plantdeconv_ct_pred", "tangram_ct_pred"]:
        if key in adata.obsm:
            pred = adata.obsm[key]
            if isinstance(pred, pd.DataFrame):
                df = pred.copy()
            else:
                df = pd.DataFrame(np.asarray(pred), index=adata.obs_names)
            df.index = adata.obs_names
            return df
    raise KeyError("Missing obsm['plantdeconv_ct_pred']")


def _infer_coordinates(adata: sc.AnnData) -> pd.DataFrame:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        return pd.DataFrame(coords[:, :2], index=adata.obs_names, columns=["coor_X", "coor_Y"])
    if "coor_X" in adata.obs.columns and "coor_Y" in adata.obs.columns:
        return adata.obs[["coor_X", "coor_Y"]].copy()
    raise KeyError("Missing spatial coordinates")


def plot_pie_map(
    adata: sc.AnnData = None,
    frac_df: pd.DataFrame = None,
    coordinates: pd.DataFrame = None,
    out_path=None,
    figsize=(14, 12),
    dpi=300,
    radius=180,
    min_slice=0.0,
    title="PlantDeconv Cell Type Pie Map",
):
    """
    Plot a pie-chart map showing cell-type proportions at each spatial spot.

    Can be called in two ways:
    1. With ``adata`` only: predictions and coordinates are inferred.
    2. With ``frac_df`` and ``coordinates`` explicitly.

    Args:
        adata (AnnData): Optional. Spatial data with predictions.
        frac_df (DataFrame): Optional. Cell-type fractions (spots x cell_types).
        coordinates (DataFrame): Optional. Columns ['coor_X', 'coor_Y'].
        out_path (str or Path): Optional. Save path. If None, plt.show() is called.
        figsize (tuple): Figure size. Default (14, 12).
        dpi (int): Resolution. Default 300.
        radius (float): Pie radius. Default 180.
        min_slice (float): Minimum fraction to draw. Default 0.0.
        title (str): Plot title.
    """
    if adata is not None:
        if frac_df is None:
            raw_pred = _infer_predict_df(adata)
            frac_df = _normalize_rows(raw_pred)
        if coordinates is None:
            coordinates = _infer_coordinates(adata)

    if frac_df is None or coordinates is None:
        raise ValueError("Either adata or both frac_df and coordinates must be provided.")

    common = coordinates.index.intersection(frac_df.index)
    coordinates = coordinates.loc[common].copy()
    frac_df = frac_df.loc[common].copy()

    x = coordinates.iloc[:, 0].to_numpy(dtype=float)
    y = coordinates.iloc[:, 1].to_numpy(dtype=float)

    celltype_order = frac_df.sum(axis=0).sort_values(ascending=False)
    frac_df = frac_df.loc[:, celltype_order.index]

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(frac_df.shape[1])]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    for i, spot in enumerate(frac_df.index):
        values = frac_df.loc[spot].to_numpy(dtype=float)
        total = values.sum()
        if total <= 0:
            continue

        start = 0.0
        drawn = False
        for j_idx, value in enumerate(values):
            if value <= min_slice:
                continue
            theta1 = start * 360.0
            theta2 = (start + value) * 360.0
            wedge = Wedge(
                center=(x[i], y[i]),
                r=radius,
                theta1=theta1,
                theta2=theta2,
                facecolor=colors[j_idx],
                edgecolor="white",
                linewidth=0.15,
            )
            ax.add_patch(wedge)
            start += value
            drawn = True

        if not drawn:
            j_idx = int(np.argmax(values))
            wedge = Wedge(
                center=(x[i], y[i]),
                r=radius,
                theta1=0.0,
                theta2=360.0,
                facecolor=colors[j_idx],
                edgecolor="white",
                linewidth=0.15,
            )
            ax.add_patch(wedge)

    pad = radius * 1.8
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=18)

    handles = [
        plt.Line2D([0], [0], color=colors[i], lw=8, label=str(frac_df.columns[i]))
        for i in range(frac_df.shape[1])
        if frac_df.iloc[:, i].sum() > 0
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, frameon=False)

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
