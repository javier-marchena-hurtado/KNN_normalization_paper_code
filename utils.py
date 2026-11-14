from types import SimpleNamespace

import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pertpy as pt
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import ScaledTranslation, blended_transform_factory
from scipy import stats
from scipy.sparse import issparse
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.utils import sparsefuncs
from mudata import MuData
import knn_normalization as knn
from statannotations.Annotator import Annotator


def pca_neighbors_umap(data, n_comps=None, n_neighbors=15):
    sc.pp.pca(data, n_comps=n_comps)
    sc.pp.neighbors(data, n_neighbors=n_neighbors)
    sc.tl.umap(data)


def neighbors_umap_leiden(data, use_PCA=True, metric="euclidean", n_neighbors=15):
    if use_PCA:
        sc.pp.neighbors(data, metric=metric, n_neighbors=n_neighbors)
    else:
        sc.pp.neighbors(data, use_rep="X", metric=metric, n_neighbors=n_neighbors)
    sc.tl.umap(data)
    sc.tl.leiden(data, flavor="igraph", n_iterations=2, directed=False)


def pca_neighbors_umap_leiden(data, n_comps=None, n_neighbors=15):
    sc.pp.pca(data, n_comps=n_comps)
    sc.pp.neighbors(data, n_neighbors=n_neighbors)
    sc.tl.umap(data)
    sc.tl.leiden(data, flavor="igraph", n_iterations=2, directed=False)


def sum_total_counts(data, obs_names=["total_counts", "log_total_counts"], log1p=False, layer=None):
    """
    Sum total per-cell counts and store them (and their logarithm) in data.obs.
    """
    if layer is None:
        total_counts = data.X.sum(axis=1)
    else:
        total_counts = data.layers[layer].sum(axis=1)
    data.obs[obs_names[0]] = total_counts
    if log1p:
        log_total_counts = np.log1p(total_counts)
    else:
        log_total_counts = np.log10(total_counts)
    data.obs[obs_names[1]] = log_total_counts


def sum_total_counts_logged(data, expm1=False, obs_names=["total_counts", "log_total_counts"], layer=None):
    """
    Sum total per-cell counts for data that has already been log-transformed:
    the exponential is taken before summing counts.
    """
    values = data.X if layer is None else data.layers[layer]
    exponential = np.expm1(values) if expm1 else np.exp(values)
    total_counts = exponential.sum(axis=1)
    data.obs[obs_names[0]] = total_counts
    data.obs[obs_names[1]] = np.log10(total_counts)


def create_protein_anndata_from_layer(data, layer_name="clr_across_cells", logged_or_raw="raw", use_PCA=True):
    """
    Create an AnnData object from a layer of the data (e.g. the CLR-normalized protein data).
    """
    # .copy() is essential: otherwise new_anndata would be a view of data and changes would propagate to it.
    new_anndata = data.copy()
    new_anndata.X = new_anndata.layers[layer_name]

    if logged_or_raw == "logged":
        sum_total_counts_logged(new_anndata)
    elif logged_or_raw == "raw":
        sum_total_counts(new_anndata)

    if use_PCA:
        sc.pp.pca(new_anndata)
    neighbors_umap_leiden(new_anndata, use_PCA=use_PCA)

    return new_anndata


def calculate_ari_bounded_max(data_samples, cell_annotation_levels, save_directory, samples_to_average, sample_word_number,
                              resolutions, selected_dataset_and_methods, methods_palette, rcparams, figsize=(2.12, 1.60),
                              profi=True, orientation="vertical", xticks=None):
    """
    Calculate the maximum ARI across clustering resolutions, restricted to resolutions whose number of clusters
    is within +-20% of the number of cell types in the annotations.
    """
    for cell_type_annotation in cell_annotation_levels:
        ari_dict = {}
        for dataset in data_samples.keys():
            if "_".join(dataset.split("_")[:sample_word_number]) in samples_to_average:
                ari_list = []
                clusterings = data_samples[dataset].obs[[f"leiden_{resolution}" for resolution in resolutions]]
                cell_type_annotations = data_samples[dataset].obs[cell_type_annotation]
                nclusters_annotations = len(cell_type_annotations.cat.categories)
                nclusters20percent = np.ceil(nclusters_annotations*0.2)
                for resolution in clusterings.columns:
                    clusters = len(clusterings[resolution].cat.categories)
                    res_number = resolution.split("_")[1]
                    ari = adjusted_rand_score(clusterings[resolution], cell_type_annotations)
                    ari_list.append([ari, res_number, clusters])
                ari_df_clustering = pd.DataFrame(ari_list, columns=["ARI", "Resolution", "n_clusters"])
                ari_bounded = ari_df_clustering[ari_df_clustering["n_clusters"].between(nclusters_annotations - nclusters20percent, nclusters_annotations + nclusters20percent, inclusive="both")]
                if ari_bounded.shape[0] == 0:
                    # No resolution within the bounded region: take the one whose number of clusters is closest to the number of cell types.
                    diff_to_nclusters_annotations = abs(ari_df_clustering["n_clusters"] - nclusters_annotations)
                    closest_resolutions_to_nclusters_annotations = ari_df_clustering[diff_to_nclusters_annotations == diff_to_nclusters_annotations.min()]
                    best_row = closest_resolutions_to_nclusters_annotations[closest_resolutions_to_nclusters_annotations["ARI"] == closest_resolutions_to_nclusters_annotations["ARI"].max()]
                    ari_max, ari_max_resolution, ari_max_clusters = float(best_row["ARI"].iloc[0]), float(best_row["Resolution"].iloc[0]), int(best_row["n_clusters"].iloc[0])
                else:
                    ari_max = float(ari_bounded["ARI"].max())
                    ari_max_resolution = float(ari_bounded[ari_bounded["ARI"] == ari_max]["Resolution"].iloc[0])
                    ari_max_clusters = int(ari_bounded[ari_bounded["ARI"] == ari_max]["n_clusters"].iloc[0])
                ari_dict[dataset] = [round(ari_max, 3), ari_max_resolution, ari_max_clusters]
        ari_df = pd.DataFrame(ari_dict, index=["ARI", "Resolution", "n_clusters"]).T
        ari_df.round(3).to_csv(f"{save_directory}/ari_scores_{cell_type_annotation}.csv", index=True)
        average_across_samples(ari_df, f"{save_directory}/ari_scores_simplified_{cell_type_annotation}.csv",
                               sample_word_number=sample_word_number)
        boxplot_comparing_normalizations(ari_df, savefilename=f"{save_directory}/ARI_{cell_type_annotation}_by_method", profi=profi, metric="ARI",
                                         selected_dataset_and_methods=selected_dataset_and_methods, sample_word_number=sample_word_number,
                                         methods_palette=methods_palette, figsize=figsize, rcparams=rcparams, orientation=orientation, xticks=xticks)


def clr_transform_matrix(matrix, axis=1, log1p=True, log_transform=True, return_geom_means=False,
                         flavor="seurat"):
    """
    Homemade function to calculate CLR.

    When using log1p, the geometric means are not strictly the geometric mean because a +1
    pseudocount has been added. But this is what CLR normalization uses.
    """
    if flavor == "seurat":
        if log1p:
            geom_means = np.exp(np.log1p(matrix).sum(axis=axis, keepdims=True) / matrix.shape[axis])
        else:
            geom_means = np.exp(np.log(matrix).sum(axis=axis, keepdims=True) / matrix.shape[axis])
        clr_transformed_matrix = matrix / geom_means

        if log_transform:
            if log1p:
                clr_transformed_matrix = np.log1p(clr_transformed_matrix)
            else:
                clr_transformed_matrix = np.log(clr_transformed_matrix)

    elif flavor == "stoeckius":
        if log1p:
            matrix += 1
        geom_means = stats.gmean(matrix, axis=axis, keepdims=True)
        transformed_matrix = matrix / geom_means
        if log_transform:
            clr_transformed_matrix = np.log(transformed_matrix, out=matrix)

    if return_geom_means:
        return clr_transformed_matrix, geom_means
    else:
        return clr_transformed_matrix


def analyze_KNN_normalized_protein(protein_knn_normalized, total_size_factors, size_factor_history, use_PCA=True):
    """
    Basic analysis of KNN-normalized protein data: add size factors to .obs/.obsm, then run PCA, neighbors, UMAP and Leiden.
    """
    protein_knn_normalized.obs["total_size_factor"] = total_size_factors
    protein_knn_normalized.obs["inverse_total_size_factor"] = 1/protein_knn_normalized.obs.total_size_factor
    protein_knn_normalized.obsm["size_factor_history"] = size_factor_history
    if use_PCA:
        sc.pp.pca(protein_knn_normalized)
    neighbors_umap_leiden(protein_knn_normalized, use_PCA=use_PCA)


def apply_normalization_methods(protein_data, rna_data=None, use_PCA_for_protein=True, verbose=True,
                                log_transform=True, pseudocount_value=5, n_neighbors=None,
                                scale_protein=True):
    """
    Applies different normalization procedures to the protein data.
    This function assumes that CLR normalized protein has been stored in a layer called "clr_across_cells".
    """

    normalized_data = {}

    if verbose:
        print("Starting sample")

    if rna_data is not None:
        sum_total_counts(rna_data)
    sum_total_counts(protein_data)

    # Calculating PCA (if necessary), neighbors, UMAPs.
    if verbose:
        print("PCA_neighbors_UMAP")
    if rna_data is not None:
        pca_neighbors_umap_leiden(rna_data)

    if use_PCA_for_protein:
        sc.pp.pca(protein_data)
    neighbors_umap_leiden(protein_data, use_PCA=use_PCA_for_protein)

    # Save AnnData objects from RNA and protein data.
    if verbose:
        print("Storing Anndata objects")
    if rna_data is not None:
        normalized_data["RNA"] = rna_data.copy()
    normalized_data["protein_raw"] = protein_data.copy()

    # Taking logarithm of protein raw data.
    if verbose:
        print("Uncorrected protein")
    protein_uncorrected = protein_data.copy()
    sc.pp.log1p(protein_uncorrected)
    if scale_protein:
        sc.pp.scale(protein_uncorrected)
    if use_PCA_for_protein:
        sc.pp.pca(protein_uncorrected)
    neighbors_umap_leiden(protein_uncorrected, use_PCA=use_PCA_for_protein)
    normalized_data["protein_uncorrected"] = protein_uncorrected.copy()

    # Library-size normalization
    if verbose:
        print("protein library-size normalized")
    library_normalization = protein_data.copy()
    library_normalization, counts_per_cell = library_normalize_total(library_normalization)
    library_normalization.obs["librarynorm_size_factors"] = counts_per_cell
    sum_total_counts(library_normalization)
    if log_transform:
        sc.pp.log1p(library_normalization)
    if scale_protein:
        sc.pp.scale(library_normalization)
    if use_PCA_for_protein:
        sc.pp.pca(library_normalization)
    neighbors_umap_leiden(library_normalization, use_PCA=use_PCA_for_protein)
    normalized_data["library_normalization"] = library_normalization.copy()

    # CLR normalization
    if verbose:
        print("protein CLR normalized")
    if log_transform:
        CLR_normalization = create_protein_anndata_from_layer(protein_data, layer_name="clr_across_cells",
                                                              use_PCA=use_PCA_for_protein, logged_or_raw="logged")
        _, geom_means = clr_transform_matrix(protein_data.X.copy(), return_geom_means=True, flavor="seurat")
        sum_total_counts_logged(CLR_normalization)
    else:
        CLR_normalization = protein_data.copy()
        CLR_normalization.X, geom_means = clr_transform_matrix(CLR_normalization.X, log_transform=False, return_geom_means=True)
        sum_total_counts(CLR_normalization)
    if scale_protein:
        sc.pp.scale(CLR_normalization)
    if use_PCA_for_protein:
        sc.pp.pca(CLR_normalization)
    neighbors_umap_leiden(CLR_normalization, use_PCA=use_PCA_for_protein)

    normalized_data["CLR_normalization"] = CLR_normalization.copy()
    normalized_data["CLR_normalization"].obs["geom_mean"] = geom_means

    # CLR normalization with stoeckius flavor
    if verbose:
        print("protein CLR normalized Stoeckius")
    if log_transform:
        CLR_normalization = create_protein_anndata_from_layer(protein_data, layer_name="clr_across_cells_stoeckius",
                                                              use_PCA=use_PCA_for_protein, logged_or_raw="logged")
        _, geom_means = clr_transform_matrix(protein_data.X.copy(), return_geom_means=True, flavor="stoeckius")
        sum_total_counts_logged(CLR_normalization)
    else:
        CLR_normalization = protein_data.copy()
        CLR_normalization.X, geom_means = clr_transform_matrix(CLR_normalization.X, log_transform=False,
                                                               return_geom_means=True, flavor="stoeckius",
                                                               log1p=False)
        sum_total_counts(CLR_normalization)
    if scale_protein:
        sc.pp.scale(CLR_normalization)
    if use_PCA_for_protein:
        sc.pp.pca(CLR_normalization)
    neighbors_umap_leiden(CLR_normalization, use_PCA=use_PCA_for_protein)

    normalized_data["CLR_normalization_stoeckius"] = CLR_normalization.copy()
    normalized_data["CLR_normalization_stoeckius"].obs["geom_mean"] = geom_means

    # CLR normalization across proteins
    if verbose:
        print("protein CLR normalized across proteins")
    if log_transform:
        CLR_normalization = create_protein_anndata_from_layer(protein_data, layer_name="clr_across_ab",
                                                              use_PCA=use_PCA_for_protein, logged_or_raw="logged")
        _, geom_means = clr_transform_matrix(protein_data.X.copy(), return_geom_means=True, axis=0, log_transform=True)
        sum_total_counts_logged(CLR_normalization)
    else:
        CLR_normalization = protein_data.copy()
        CLR_normalization.X, geom_means = clr_transform_matrix(CLR_normalization.X, log_transform=False,
                                                               return_geom_means=True, axis=0)
        sum_total_counts(CLR_normalization)
    if scale_protein:
        sc.pp.scale(CLR_normalization)
    if use_PCA_for_protein:
        sc.pp.pca(CLR_normalization)
    neighbors_umap_leiden(CLR_normalization, use_PCA=use_PCA_for_protein)

    normalized_data["CLR_normalization_across_ab"] = CLR_normalization.copy()
    normalized_data["CLR_normalization_across_ab"].var["geom_mean"] = geom_means.flatten()

    # KNN normalization with RNA neighbors
    if rna_data is not None:
        if verbose:
            print("KNN normalization with RNA neighbors")

        mudata = MuData({"prot": protein_data, "rna": rna_data})
        mudata_knn_normalized = knn.knn_normalize(mudata, calculate_neighbors_from="rna",
                                                    log_transform=log_transform, n_neighbors=n_neighbors, pseudocount=pseudocount_value,
                                                    inplace=False, save_size_factors=True, verbose=verbose)
        protein_knn_normalized = mudata_knn_normalized["prot"]

        sum_total_counts_logged(protein_knn_normalized)
        if scale_protein:
            sc.pp.scale(protein_knn_normalized)
        analyze_KNN_normalized_protein(protein_knn_normalized, protein_knn_normalized.obs["size_factor"], protein_knn_normalized.obsm["size_factor_history"], use_PCA=use_PCA_for_protein)
        normalized_data["KNN_RNA_neighbors"] = protein_knn_normalized.copy()

    # KNN normalization with protein neighbors
    if verbose:
        print("KNN normalization with protein neighbors")

    protein_knn_normalized_cosine = knn.knn_normalize(protein_data, calculate_neighbors_from="prot", log_transform=log_transform,
                                                      n_neighbors=n_neighbors, pseudocount=pseudocount_value, inplace=False,
                                                      save_size_factors=True, verbose=verbose)
    sum_total_counts_logged(protein_knn_normalized_cosine)
    if scale_protein:
        sc.pp.scale(protein_knn_normalized_cosine)
    analyze_KNN_normalized_protein(protein_knn_normalized_cosine, protein_knn_normalized_cosine.obs["size_factor"], protein_knn_normalized_cosine.obsm["size_factor_history"], use_PCA=use_PCA_for_protein)
    normalized_data["KNN_protein_neighbors"] = protein_knn_normalized_cosine.copy()

    return normalized_data


def normalize_per_sample(protein_data, rna_data=None, sample_list=None, sample_column_name="sample", use_PCA_for_protein=True, verbose=True,
                         log_transform=True, pseudocount_value=5, n_neighbors=None, scale_protein=True):
    """
    Applies different normalization procedures to each sample separately. Returns a dictionary
    "data_samples" which contains all the normalized data.

    sample_list: names of the samples for which the normalization procedure should be applied.
    sample_column_name: name of the column in data.obs that indicates the sample to which each cell belongs.
    """

    data_samples = {}
    for sample in sample_list:
        print(sample)
        # Select the data from the sample.
        if rna_data is not None:
            sample_rna = rna_data[rna_data.obs[sample_column_name] == sample].copy()
        else:
            sample_rna = None
        sample_protein = protein_data[protein_data.obs[sample_column_name] == sample].copy()

        # Apply the different normalization methods to the protein data.
        normalized_data = apply_normalization_methods(protein_data=sample_protein, rna_data=sample_rna, use_PCA_for_protein=use_PCA_for_protein,
                                                      verbose=verbose, log_transform=log_transform,
                                                      pseudocount_value=pseudocount_value, n_neighbors=n_neighbors, scale_protein=scale_protein)
        # Prefix the sample name and add to the dictionary with all the samples.
        data_samples.update({f"{sample}_{key}": value for key, value in normalized_data.items()})

    return data_samples


def average_across_samples(df, savefilename, sample_word_number=1):
    """
    Average a dataframe across samples. The assay name is assumed to be the index of the dataframe, and the first words
    of the assay name are the sample.
    """
    # Group by everything except the first words (the sample), then take the mean across samples.
    levels = df.index.str.split('_').str[sample_word_number:].str.join('_')
    simplified_df = df.groupby(levels, sort=False).mean()
    simplified_df.round(3).to_csv(savefilename)

    return simplified_df


def library_normalize_total(anndata, target_sum=None):
    """
    Library-size normalization. Based on sc.pp.normalize_total().
    """
    X = anndata.X
    if issubclass(X.dtype.type, (int, np.integer)):
        X = X.astype(np.float32)

    counts_per_cell = X.sum(1)
    counts_per_cell = np.ravel(counts_per_cell)
    counts_greater_than_zero = counts_per_cell[counts_per_cell > 0]
    target_sum = np.median(counts_greater_than_zero, axis=0) if target_sum is None else target_sum
    counts_per_cell += counts_per_cell == 0
    counts_per_cell = counts_per_cell / target_sum
    if issparse(X):
        sparsefuncs.inplace_row_scale(X, 1 / counts_per_cell)
    elif isinstance(counts_per_cell, np.ndarray):
        np.divide(X, counts_per_cell[:, None], out=X)
    else:
        X = np.divide(X, counts_per_cell[:, None])  # dask does not support kwarg "out"
    anndata.X = X
    return anndata, counts_per_cell


def boxplot_comparing_normalizations(df, savefilename, methods_palette, sample_word_number, selected_dataset_and_methods,
                                     profi=True, metric="SNR", figsize=(12, 6),
                                     rcparams={'font.size': 19, 'axes.titlesize': 22, 'axes.labelsize': 16, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14, 'figure.titlesize': 22},
                                     orientation="vertical", xticks=None):
    """Boxplot to compare the ARI, silhouette, SNR, etc. by normalization method."""

    df["sample"] = df.index.str.split("_").str[:sample_word_number].str.join("_")
    df["normalization"] = df.index.str.split("_").str[sample_word_number:].str.join("_")
    if profi:
        df = df[df["normalization"].isin(selected_dataset_and_methods.keys())].copy()
    df["normalization"] = df["normalization"].replace(selected_dataset_and_methods)

    if metric == "SNR":
        df = df.melt(id_vars=["normalization", "sample"], value_vars=df.columns[1:-2], var_name="Marker", value_name="SNR")

    plt.rcParams.update(rcparams)
    plt.figure(figsize=figsize)

    if profi:
        # Order of categories and their colors for consistent plotting.
        method_order = list(selected_dataset_and_methods.values())
        palette = {method: methods_palette[method] for method in method_order if method in methods_palette}
        if orientation == "vertical":
            sns.boxplot(x="normalization", y=metric, data=df, order=method_order, palette=palette, hue="normalization")
        else:
            sns.boxplot(x=metric, y="normalization", data=df, order=method_order, palette=palette, hue="normalization", orientation="horizontal")
    else:
        if orientation == "vertical":
            sns.boxplot(x="normalization", y=metric, data=df, hue="normalization")
        else:
            sns.boxplot(x=metric, y="normalization", data=df, hue="normalization", orientation="horizontal")
    plt.xticks(rotation=90)
    dict_metric_name = {"SNR": "Signal to Noise Ratio", "ari": "Adjusted Rand Index", "ARI": "Adjusted Rand Index", "nmi": "Normalized Mutual Information", "silhouette": "Silhouette score", "knn_classifier_accuracy": "KNN Classifier Accuracy"}
    if orientation == "vertical":
        plt.ylabel(dict_metric_name[metric])
        plt.xlabel("")
    else:
        plt.ylabel("")
        plt.xlabel(dict_metric_name[metric])
    if xticks:
        plt.xticks(xticks)
    plt.savefig(f"{savefilename}.svg", dpi=300)
    plt.savefig(f"{savefilename}.png", dpi=300, bbox_inches="tight")
    plt.close("all")


def compare_size_factors(
    anndata,
    estimated_size_factor,
    plot_title,
    saveplotname,
    palette,
    point_size=25,
    true_size_factor="true_size_factor",
    regression_lines=False,
    xticks=None,
    yticks=None,
):
    """
    Creates plots that compare the true size factors to the calculated size factors.

    Parameters
    ----------
    xticks : list-like or None
        Tick locations for the x-axis.
    yticks : list-like or None
        Tick locations for the y-axis.
    """

    obs = anndata.obs.copy()
    obs = obs.sample(frac=1, random_state=42).reset_index(drop=True)

    x, y = obs[true_size_factor], obs[estimated_size_factor]
    xlim, ylim = np.percentile(x, 99.5), np.percentile(y, 99.5)

    fig, ax = plt.subplots(figsize=(2.12, 1.60), dpi=300)

    sns.scatterplot(
        data=obs,
        x=true_size_factor,
        y=estimated_size_factor,
        hue="Cell type",
        s=point_size,
        marker=".",
        palette=palette,
        ax=ax,
        legend=False,
    )

    ax.set_xlabel("True size factor")
    ax.set_ylabel("Estimated size factor")
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    if regression_lines:
        for cell_type in obs["Cell type"].unique():
            data_cell_type = obs[obs["Cell type"] == cell_type]

            X = sm.add_constant(data_cell_type[true_size_factor].values)
            reg = sm.OLS(data_cell_type[estimated_size_factor], X).fit()

            intercept, coefficient = reg.params

            reg_x = np.linspace(0, xlim, 100)
            reg_y = intercept + coefficient * reg_x

            ax.plot(reg_x, reg_y, color="red", linewidth=0.8)

    ax.set_title(plot_title, fontsize=plt.rcParams["axes.labelsize"])

    plt.savefig(saveplotname, bbox_inches="tight")
    plt.close("all")


def remove_very_high_total_counts_by_sample(protein_data, rna_data, sample_column_name, percentile):
    """
    Remove, for each sample, the cells with the highest total protein counts (above the given percentile).
    """
    for sample in protein_data.obs[sample_column_name].unique():
        counts_percentile = np.percentile(protein_data[protein_data.obs[sample_column_name] == sample].obs["total_counts"], percentile)
        cells_to_remove = (protein_data.obs[sample_column_name] == sample) & (protein_data.obs["total_counts"] >= counts_percentile)
        protein_data = protein_data[~cells_to_remove].copy()
        rna_data = rna_data[~cells_to_remove].copy()
    return protein_data, rna_data


# ---------------------------------------------------------------------------
# Figure 4 (CITE-seq CRC annotation)
# ---------------------------------------------------------------------------
BASE_DIR = "/groups/nils/members/javier/Immune_colon/annotation_CITE-seq_CRC"
DATA_DIR, FIG_DIR = f"{BASE_DIR}/data", f"{BASE_DIR}/figures"
L1, L2 = "annotation_combined_v1_l1", "annotation_combined_v1_l2"

CELL_TYPES_PALETTE = {
    "Treg": "#B5DEAD", "CD4 T": "#F75C03", "CD8 T": "#157F1F", "innate CD8 T": "#76B041",
    "NK": "#B2ABF2", "CD16 NK": "#2708A0", "CD16- NK": "#CCE6F4", "B": "#6EA4BF",
    "naive/transitional B": "#48ACF0", "germinal B": "#3A5683", "Plasma": "#1A181B", "DC": "#CBC0AD",
    "cDC": "#FFD166", "pDC": "#BA2D0B", "DC, activated": "#FFCAD4", "Macro": "#15616D",
    "Mono-Macro": "#D8DDEF", "Mono": "#A833B9", "Mast": "#7B2D26", "Myeloid": "#2708A0"}
MARKERS = ["CD4", "CD5", "CD3", "CD95", "CD278", "CD25", "CD8a", "CD103", "CD335",
           "CD20", "CD19", "CD21", "CD27", "CD38", "CD39", "CD33", "CD11c"]
SIMPLE_MAPPING = {"CD4 T": "CD4 T & Treg", "Treg": "CD4 T & Treg", "CD8 T": "CD8 T", "Myeloid": "Myeloid"}
SIMPLE_COLORS = ["#F75C03", "#157F1F", "#B2BEB5", "#2708A0"]
TUMOR_ADJACENT_PALETTE = {"tumor": "#E9A820", "adjacent": "#2A9D8F"}

SCCODA_MIN_CELLS = 65
SCCODA_GROUPS = {
    "T cells": ["CD4 T", "Treg", "CD8 T"],
    "NK cells": ["CD16 NK", "CD16- NK"],
    "B cells": ["naive/transitional B", "germinal B"],
    "Plasma cells": ["Plasma IgA+", "Plasma IgG+", "Plasma IgA+ IgG+"],
}

# (cell type, genes) blocks of the DESeq heatmap, per annotation column the DESeq CSVs were computed with
DESEQ_BLOCKS = {
    L1: [("Plasma", ["CCND2", "IGHG1", "IGHG2", "IGHG3", "HSPA1A"]),
         ("CD8 T", ["CXCL13", "CCL3", "GZMK", "GZMB", "CTLA4", "HLA-DRA", "FABP5", "DNAJB1", "SERPINH1", "HSPH1",
                    "HSP90AA1", "BAG3", "CD82", "CD27", "RBPJ", "NR4A2", "SPRY1"]),
         ("CD4 T", ["CXCL13", "TNFRSF18", "DUSP4", "HSPA1B", "HSPH1", "HSP90AA1", "HSPD1", "RPS29", "AL138963.3"])],
    "Smooth_cell_type": [
        ("Plasma", ["GNLY", "IGHG1", "CCND2", "IGHG3", "NKG7", "LTB", "FYB1", "CD2", "CD3E", "CD3D", "IGHG4", "CXCR4",
                    "HLA-DRB1", "IL32", "TAGAP", "RHOB", "JCHAIN", "TIMP1", "RPL36A", "CCL3", "GPR15", "AMPD1"]),
        ("CD8 T", ["CXCL13", "HSPA1A", "SERPINH1", "CTLA4", "ZFAND2A", "CD82", "CD27", "FABP5", "RBPJ", "DNAJB1",
                   "BATF", "GZMK", "HSPH1", "HLA-DRA", "HSPB1", "HSP90AA1", "HOPX", "LINC01871", "CCL4L2", "MAP3K8",
                   "KIR2DL4", "NR4A2", "XCL2", "PTGER2", "XCL1"]),
        ("CD4 T", ["CXCL13", "HSPA1B", "HSP90AA1"])],
}

SIGNATURES = {"literature_cytotoxicity_signature": "Cytotoxicity", "literature_exhaustion_signature": "Exhaustion"}
SIGNATURE_COLORS = {"Cytotoxicity": "#d6604d", "Exhaustion": "#4393c3"}

FIG_W_IN, FIG_H_IN = 178 / 25.4, 8.6   # Bioinformatics double-column width
ROW_HEIGHTS = [1.6, 1.0, 1.05, 1.5]    # rows: A|B, C, D, E|F
FS = 8                                 # one font size for all text
FIG_RCPARAMS = {
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "legend.title_fontsize": FS, "figure.titlesize": FS,
    "text.usetex": False, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 0.8, "patch.linewidth": 0.5,
    "pdf.fonttype": 42,  # editable text in the PDF
}


def load_data():
    protein = sc.read_h5ad(f"{DATA_DIR}/CRC_CITE-seq_batch_corrected_adata_v35.h5ad")
    rna = sc.read_h5ad(f"{DATA_DIR}/rna_v15.h5ad")
    return protein, rna


def add_simplified_annotations(protein):
    """Add the simplified CITE-seq / scRNA-seq annotations used in panel B. Returns (order, {category: color})."""
    for new, src in [("protein_annotation_simplified", L1), ("scrnaseq_annotation_simplified", "Smooth_cell_type")]:
        protein.obs[new] = protein.obs[src].astype(str).map(SIMPLE_MAPPING).fillna("Others").astype("category")
        protein.uns[f"{new}_colors"] = SIMPLE_COLORS
    col = "protein_annotation_simplified"
    protein.obs[col] = protein.obs[col].cat.reorder_categories(["CD4 T & Treg", "CD8 T", "Others", "Myeloid"])
    order = list(protein.obs[col].cat.categories)
    return order, dict(zip(order, protein.uns[f"{col}_colors"]))


def get_values(adata, name):
    """Values of an obs column or of a gene/protein."""
    if name in adata.obs.columns:
        return adata.obs[name].values
    if name not in adata.var_names:
        raise ValueError(f"{name} not found in obs or var_names")
    X = adata[:, name].X
    return np.asarray(X.toarray() if issparse(X) else X).ravel()


def prepare_sccoda(protein):
    """scCODA object without the L2 cell types that have fewer than SCCODA_MIN_CELLS cells."""
    counts = protein.obs[L2].value_counts()
    sub = protein[protein.obs[L2].isin(counts[counts >= SCCODA_MIN_CELLS].index)].copy()
    sub.obs[L2] = sub.obs[L2].cat.remove_unused_categories()
    return pt.tl.Sccoda().load(sub, type="cell_level", generate_sample_level=True, cell_type_identifier=L2,
                               sample_identifier="sample", covariate_obs=["origin"])


def sccoda_long_table(sccoda_data):
    """Relative abundance per sample, long format. Returns (df_long, cell_types)."""
    coda = sccoda_data["coda"]
    cell_cols = list(coda.var_names)
    X = coda.X.toarray() if issparse(coda.X) else coda.X
    df = pd.DataFrame(X, index=coda.obs_names, columns=cell_cols)
    df = df.div(df.sum(axis=1), axis=0)
    df["origin"] = coda.obs["origin"].values
    return df.melt(id_vars="origin", var_name="cell_type", value_name="rel_abundance"), cell_cols


def marker_means_by_cell_type(protein):
    """Mean protein expression of MARKERS per L1 cell type (pDC excluded)."""
    pm = protein[protein.obs[L1] != "pDC", MARKERS]
    X = pm.X.toarray() if issparse(pm.X) else np.asarray(pm.X)
    df = pd.DataFrame(X, index=pm.obs_names, columns=pm.var_names)
    return df.groupby(pm.obs[L1].values, observed=True).mean()


def prepare_deseq_heatmap_data(protein, annotation=L1):
    """Per-gene z-scored log2 DESeq-normalized matrix: tumor samples, a blank row, adjacent samples (rows) by
    the genes of each DESEQ_BLOCKS cell type separated by blank columns. Returns (mat, labels, spans, rows, tumor, adjacent)."""
    origin = protein.obs[["sample", "origin"]].astype(str).drop_duplicates().set_index("sample")["origin"].str.lower()
    tumor, adjacent = sorted(origin[origin == "tumor"].index), sorted(origin[origin == "adjacent"].index)
    rows = tumor + [" "] + adjacent

    mat, labels, spans = [], [], []
    for cell_type, genes in DESEQ_BLOCKS[annotation]:
        df = pd.read_csv(f"{DATA_DIR}/DESeq-normalized_{cell_type}_{annotation}_origin_rna.csv", index_col=0)
        # pseudobulk index is "{sample}_{origin}": strip the suffix
        sfx = "_" + df["origin"].astype(str)
        df.index = [i[:-len(s)] if i.endswith(s) else i for i, s in zip(df.index, sfx)]
        m = np.log2(df.drop(columns="origin", errors="ignore").reindex(columns=genes) + 1).reindex(rows)
        mat += [((m - m.mean()) / m.std()).values, np.full((len(rows), 1), np.nan)]
        spans.append((len(labels), len(labels) + len(genes), cell_type))
        labels += genes + [""]
    return np.hstack(mat[:-1]), labels[:-1], spans, rows, tumor, adjacent


def prepare_t_cells_tcr(protein):
    """Treg / CD4 T / CD8 T cells with a detected TCR."""
    t = protein[protein.obs[L1].isin(["Treg", "CD4 T", "CD8 T"])]
    return t[t.obs["clonal_expansion"] != "no_TCR_detected"].copy()


def prepare_rna_tumor_t_cells(rna):
    """Tumor T cells with a detected TCR from samples p026t and p042t."""
    t = rna[(rna.obs["origin"] == "tumor") & rna.obs[L1].isin(["Treg", "CD8 T", "CD4 T"])]
    t = t[(t.obs["clonal_expansion"] != "no_TCR_detected") & t.obs["sample"].isin(["p026t", "p042t"])].copy()
    t.uns["clonal_expansion_colors"] = ["#A9A3AC", "#C47FBD", "#6E2266"]
    return t


def prepare_figure_data(protein, rna):
    order, colors = add_simplified_annotations(protein)
    return SimpleNamespace(simple_order=order, simple_colors=colors, sccoda_data=prepare_sccoda(protein),
                           t_cells_tcr=prepare_t_cells_tcr(protein), rna_t=prepare_rna_tumor_t_cells(rna))


def _hide_blank_ticks(ax, x_labels, y_labels):
    for axis, labs in [(ax.xaxis, x_labels), (ax.yaxis, y_labels)]:
        for tick, lab in zip(axis.get_major_ticks(), labs):
            if not lab.strip():
                tick.tick1line.set_visible(False)


def _label_stacked_pct(ax, comp, colors, fontsize=FS, min_pct=5, only=None, rotation=0):
    """Percentage inside each stacked segment (>= min_pct); `only` restricts it to one category."""
    for x, (_, row) in enumerate(comp.T.iterrows()):
        bottom = 0
        for cat, val, col in zip(row.index, row.values, colors):
            if val >= min_pct and only in (None, cat):
                r, g, b = to_rgb(col)
                ax.text(x, bottom + val / 2, f"{val:.0f}%", ha="center", va="center", fontsize=fontsize,
                        rotation=rotation, color="black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white")
            bottom += val


def panel_A(ax, protein):
    df = marker_means_by_cell_type(protein)
    sns.heatmap(df, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"ticks": [-1, 0, 1], "pad": 0.03, "fraction": 0.06})
    ax.set(xlabel="", ylabel="", title="CITE-seq-based cell type annotations")
    ax.set_xticks(np.arange(df.shape[1]) + 0.5)
    ax.set_xticklabels(df.columns, rotation=90, ha="center")
    ax.set_yticks(np.arange(df.shape[0]) + 0.5)
    ax.set_yticklabels(df.index, rotation=0, ha="right", fontweight="bold")
    for t in ax.get_yticklabels():
        t.set_color(CELL_TYPES_PALETTE.get(t.get_text(), "black"))
    ax.tick_params(length=0)
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(["<-1", "0", ">1"])  # values are clipped at vmin/vmax
    cbar.ax.tick_params(labelsize=FS, length=2)
    cbar.set_label("Mean scaled \nprotein expression", labelpad=2)
    cbar.outline.set_visible(False)


def panel_B(fig, placeholder, protein, d, left_pad=0.38, gap=0.30, bar_w=0.45, bar_right=0.30, leg_x=0.36,
            leg_w=0.68, leg_up=0.12, extend_top=0.0, extend_bottom=0.06, mid_shift=0.06, connector_pad=0.08):
    """CD4 vs CD8a scatter with the gated CD8 T / CD4 T & Treg regions and stacked bars of their composition,
    placed by hand (all lengths in inches) inside the `placeholder` axes. Returns (scatter, CD8 bar, CD4 bar) axes."""
    order, colors = d.simple_order, d.simple_colors
    W, H = FIG_W_IN, FIG_H_IN
    p = placeholder.get_position()
    x0, y0, w, h = p.x0 * W, p.y0 * H, p.width * W, p.height * H
    placeholder.remove()

    def ax_in(l, b, ww, hh):
        return fig.add_axes([l / W, b / H, ww / W, hh / H])

    # gating
    cd4, cd8 = get_values(protein, "CD4"), get_values(protein, "CD8a")
    c4_x, c4_y = 0.4, 0.5   # CD4 T region: CD4 > c4_x, CD8a < c4_y
    c8_x, c8_y = 0.2, 0.7   # CD8 T region: CD4 < c8_x, CD8a > c8_y
    mask_cd8, mask_cd4 = (cd8 > c8_y) & (cd4 < c8_x), (cd8 < c4_y) & (cd4 > c4_x)
    xlim, ylim = (-2.3, 3), (-2.5, 3.6)

    # axes positions
    sc_left = x0 + left_pad
    sc_w = w - left_pad - gap - bar_w - max(bar_right, leg_x + leg_w)
    bar_x = sc_left + sc_w + gap

    def y_in(v):  # scatter data y -> figure inches
        return y0 + (v - ylim[0]) / (ylim[1] - ylim[0]) * h

    # the lower bar's title needs vertical room: pull the bars in at the middle if necessary
    cd4_title = "CD4 T &\nTreg region"
    title_room = (FS * 1.25 * 2 + plt.rcParams["axes.titlepad"] + 3) / 72
    shift = max(mid_shift, (title_room - (y_in(c8_y) - y_in(c4_y))) / 2)
    cd8_top, cd8_bottom = y0 + h + extend_top, y_in(c8_y) + shift
    cd4_top, cd4_bottom = y_in(c4_y) - shift, y0 - extend_bottom
    offset = (y_in(c8_y) + y_in(c4_y)) / 2 - (cd8_bottom + cd4_top) / 2   # keep the gap centered on the corners
    cd8_bottom, cd4_top = cd8_bottom + offset, cd4_top + offset

    ax_sc = ax_in(sc_left, y0, sc_w, h)
    ax_cd8 = ax_in(bar_x, cd8_bottom, bar_w, cd8_top - cd8_bottom)
    ax_cd4 = ax_in(bar_x, cd4_bottom, bar_w, cd4_top - cd4_bottom)
    ax_leg = ax_in(bar_x + bar_w + leg_x, y0, leg_w, cd8_top - y0 + leg_up)
    ax_leg.axis("off")

    # scatter
    pt_colors = protein.obs["protein_annotation_simplified"].astype(str).map(colors).values
    ax_sc.scatter(cd4, cd8, s=0.8, c=pt_colors, linewidths=0, alpha=0.6, rasterized=True)
    ax_sc.set_xlabel("CD4")
    ax_sc.set_ylabel("CD8a", labelpad=-3)
    ax_sc.set(xlim=xlim, ylim=ylim)
    # dashed gates drawn unclipped; a white underlay on the edges lying on the spines lets the dashes show
    rect = dict(linewidth=0.8, edgecolor="black", facecolor="none", linestyle="--", clip_on=False, zorder=5)
    ax_sc.add_patch(patches.Rectangle((xlim[0], c8_y), c8_x - xlim[0], ylim[1] - c8_y, **rect))
    ax_sc.add_patch(patches.Rectangle((c4_x, ylim[0]), xlim[1] - c4_x, c4_y - ylim[0], **rect))
    under = dict(color="white", linewidth=0.8, solid_capstyle="butt", clip_on=False, zorder=4)
    for xs, ys in [([c4_x, xlim[1]], [ylim[0]] * 2), ([xlim[1]] * 2, [ylim[0], c4_y]),
                   ([xlim[0]] * 2, [c8_y, ylim[1]]), ([xlim[0], c8_x], [ylim[1]] * 2)]:
        ax_sc.plot(xs, ys, **under)

    # stacked bars: CITE vs RNA annotation of the cells in each region
    bar_colors = [colors[c] for c in order]

    def stacked_bar(ax, mask, title, only, show_x):
        pct = lambda col: protein.obs[col][mask].value_counts(normalize=True).reindex(order).fillna(0) * 100
        comp = pd.concat([pct("protein_annotation_simplified").rename("CITE"),
                          pct("scrnaseq_annotation_simplified").rename("RNA")], axis=1)
        comp.T.plot(kind="bar", stacked=True, ax=ax, width=0.7, legend=False, edgecolor="none", color=bar_colors)
        _label_stacked_pct(ax, comp, bar_colors, only=only, rotation=90)
        ax.set(ylim=(0, 100), yticks=[0, 50, 100], title=title, xlabel="", ylabel="% of cells" if show_x else "")
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        if show_x:
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
            for lab, dx in zip(ax.get_xticklabels(), (-3, 3)):   # nudge CITE / RNA apart
                lab.set_transform(lab.get_transform() + ScaledTranslation(dx / 72, 0, fig.dpi_scale_trans))
            ax.set_xlabel("Annotation")
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)

    stacked_bar(ax_cd8, mask_cd8, "CD8 T region", "CD8 T", show_x=False)
    stacked_bar(ax_cd4, mask_cd4, cd4_title, "CD4 T & Treg", show_x=True)

    # connectors from each region's outer corners to its bar
    for top, bottom, bar_ax in [((c8_x, ylim[1]), (c8_x, c8_y), ax_cd8), ((xlim[1], c4_y), (xlim[1], ylim[0]), ax_cd4)]:
        for corner, yb in [(top, 1.0), (bottom, 0.0)]:
            fig.add_artist(ConnectionPatch(xyA=corner, coordsA="data", axesA=ax_sc, xyB=(-connector_pad, yb),
                                           coordsB="axes fraction", axesB=bar_ax, color="gray", linestyle="--",
                                           linewidth=0.6, clip_on=False))

    ax_leg.legend(handles=[Patch(facecolor=colors[c], label=c) for c in order], title="Cell type", loc="upper left",
                  bbox_to_anchor=(0, 1), frameon=False, handlelength=0.8, handleheight=1, handletextpad=0.4,
                  borderpad=0.2, borderaxespad=0)
    return ax_sc, ax_cd8, ax_cd4


def panel_C(ax, d):
    df_long, cell_cols = sccoda_long_table(d.sccoda_data)
    kw = dict(data=df_long, x="cell_type", y="rel_abundance", hue="origin", ax=ax)
    sns.boxplot(**kw, showfliers=False, linewidth=0.6, width=0.7, palette=TUMOR_ADJACENT_PALETTE)
    sns.stripplot(**kw, dodge=True, palette=["black"] * 2, size=2, jitter=True, legend=False)
    ax.set_xlabel("")
    ax.set_ylabel("% of CD45+ cells in sample", y=0.3)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    pairs = [((c, "adjacent"), (c, "tumor")) for c in ["Treg", "Plasma IgA+", "CD8 T"]]
    annot = Annotator(ax, pairs, data=df_long, x="cell_type", y="rel_abundance", hue="origin")
    annot.configure(test=None, text_format="star", loc="inside", verbose=0, fontsize=FS, line_width=0.6)
    annot.set_custom_annotations(["**", "**", "*"])
    annot.annotate()

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], frameon=False, loc="upper right")
    ax.set_title("Cell type composition per sample")
    return cell_cols


def add_group_brackets(ax, groups, order, renderer, gap_pt=3, tick_pt=3):
    """Brackets a fixed number of points below the lowest x tick label (stable when the layout is resized)."""
    fig = ax.figure
    lowest = min(t.get_window_extent(renderer).y0 for t in ax.get_xticklabels() if t.get_text())
    depth_pt = (ax.transAxes.transform((0, 0))[1] - lowest) * 72 / fig.dpi + gap_pt
    tr = blended_transform_factory(ax.transData, ax.transAxes) + ScaledTranslation(0, -depth_pt / 72, fig.dpi_scale_trans)
    line = dict(arrowstyle="-", lw=0.8, color="black", shrinkA=0, shrinkB=0)
    for label, members in groups.items():
        pos = [order.index(m) for m in members if m in order]
        if not pos:
            continue
        x0, x1 = min(pos) - 0.3, max(pos) + 0.3
        ax.annotate("", xy=(x0, 0), xycoords=tr, xytext=(x1, 0), textcoords=tr, arrowprops=line, annotation_clip=False)
        for x in (x0, x1):
            ax.annotate("", xy=(x, 0), xycoords=tr, xytext=(0, tick_pt), textcoords="offset points",
                        arrowprops=line, annotation_clip=False)
        ax.annotate(label, xy=((x0 + x1) / 2, 0), xycoords=tr, xytext=(0, -2), textcoords="offset points",
                    ha="center", va="top", fontsize=FS, annotation_clip=False)


def panel_D(ax, protein, annotation=L1):
    mat, labels, spans, rows, tumor, adjacent = prepare_deseq_heatmap_data(protein, annotation)
    sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=-2, vmax=2, xticklabels=labels, yticklabels=rows,
                linewidths=0.3, linecolor="white", ax=ax,
                cbar_kws={"pad": 0.015, "fraction": 0.025, "ticks": [-2, -1, 0, 1, 2]})
    cbar = ax.collections[0].colorbar
    cbar.set_label("Z-scored log2\n(DESeq-normalized counts)", fontsize=FS)
    cbar.ax.tick_params(labelsize=FS, length=2)
    cbar.outline.set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=90, fontstyle="italic")
    plt.setp(ax.get_yticklabels(), rotation=0)
    _hide_blank_ticks(ax, labels, rows)

    for x0, x1, cell_type in spans:
        ax.plot([x0 + .1, x0 + .1, x1 - .1, x1 - .1], [-0.2, -0.5, -0.5, -0.2], color="k", lw=0.8, clip_on=False)
        ax.text((x0 + x1) / 2, -0.65, cell_type, ha="center", va="bottom", fontsize=FS, fontweight="bold")
    for txt, yc in [("Tumor", len(tumor) / 2), ("Adjacent", len(tumor) + 1 + len(adjacent) / 2)]:
        ax.annotate(txt, xy=(0, yc), xycoords=("axes fraction", "data"), xytext=(-42, 0), textcoords="offset points",
                    rotation=90, va="center", ha="center", fontweight="bold", fontsize=FS)
    ax.set_title("Differentially expressed genes, CRC vs adjacent normal colon", pad=20)


def panel_E(ax, d):
    t = d.t_cells_tcr
    tab = pd.crosstab(t.obs[L1], t.obs["clonal_expansion"], normalize="index")
    tab = tab.reindex(columns=t.obs["clonal_expansion"].cat.categories)
    tab.plot(kind="bar", stacked=True, color=t.uns["clonal_expansion_colors"], width=0.75, edgecolor="none",
             ax=ax, legend=False)
    ax.set(xlabel="", ylabel="Fraction of T cells", title="Tumor T cell clonal expansion")
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Clonal\nexpansion", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
              handlelength=1, handleheight=1)


def panel_F(ax, ax_leg, adata, signatures=SIGNATURES, signature_colors=SIGNATURE_COLORS,
            title="Tumor T cells by clonal expansion", ylabel="Signature score", yticks=None, connect_samples=False):
    """Signature scores by clonal expansion: violins + single cells, sample means as colored dots."""
    exp_order, sig_order = list(adata.obs["clonal_expansion"].cat.categories), list(signatures.values())
    long = adata.obs[["sample", "clonal_expansion", *signatures]].melt(
        id_vars=["sample", "clonal_expansion"], var_name="signature", value_name="score")
    long["signature"] = long["signature"].map(signatures)
    avg = long.groupby(["sample", "clonal_expansion", "signature"], observed=True, as_index=False)["score"].mean()

    sample_cols = dict(zip(adata.obs["sample"].cat.categories, adata.uns["sample_colors"]))
    samples = [s for s in sample_cols if s in set(avg["sample"])]
    kw = dict(data=long, x="clonal_expansion", y="score", hue="signature", order=exp_order,
              hue_order=sig_order, legend=False, ax=ax)

    sns.violinplot(**kw, palette=signature_colors, inner=None, cut=0, density_norm="width", linewidth=0.6)
    for coll in ax.collections:
        coll.set_alpha(0.45)
    n_violin = len(ax.collections)
    sns.stripplot(**kw, palette={s: "black" for s in sig_order}, dodge=True, jitter=0.15, size=0.6)
    for coll in ax.collections[n_violin:]:
        if isinstance(coll, PathCollection):
            coll.set_rasterized(True)   # keeps the PDF light

    # sample means, dodged by signature and (slightly) by sample
    sig_off = {s: -0.4 + 0.8 / len(sig_order) * (j + 0.5) for j, s in enumerate(sig_order)}
    samp_off = dict(zip(samples, np.linspace(-0.05, 0.05, len(samples)))) if len(samples) > 1 else {samples[0]: 0}
    for sample in samples:
        for sig in sig_order:
            dd = (avg[(avg["sample"] == sample) & (avg["signature"] == sig)]
                  .set_index("clonal_expansion").reindex(exp_order).dropna(subset=["score"]))
            xs = [exp_order.index(e) + sig_off[sig] + samp_off[sample] for e in dd.index]
            if connect_samples:
                ax.plot(xs, dd["score"], color=sample_cols[sample], lw=0.8, alpha=0.7, zorder=9)
            ax.scatter(xs, dd["score"], s=22, color=sample_cols[sample], edgecolor="black", linewidth=0.5, zorder=10)

    ax.set(title=title, xlabel="", ylabel=ylabel, xticks=range(len(exp_order)))
    ax.set_xticklabels(exp_order)
    if yticks is not None:
        ax.set_yticks(yticks)

    sig_handles = [Patch(facecolor=signature_colors[s], alpha=0.45, edgecolor="black", label=s) for s in sig_order]
    sample_handles = [Line2D([], [], marker="o", ls="", markersize=5, markerfacecolor=sample_cols[s],
                             markeredgecolor="black", markeredgewidth=0.5, label=s) for s in samples]
    leg = ax_leg.legend(handles=sig_handles, title="Signature", loc="upper left", bbox_to_anchor=(0, 1),
                        frameon=False, borderaxespad=0, handlelength=1, handleheight=1)
    ax_leg.add_artist(leg)
    ax_leg.legend(handles=sample_handles, title="Sample mean", loc="upper left", bbox_to_anchor=(0, 0.5),
                  frameon=False, borderaxespad=0)


def make_figure_4(protein, rna, save=True, fig_dir=FIG_DIR, fig_name="figure_4_v0"):
    """Multi-panel Figure 4 (A-F); saves PNG + PDF to `fig_dir`."""
    d = prepare_figure_data(protein, rna)
    plt.rcdefaults()
    plt.rcParams.update(FIG_RCPARAMS)

    # constrained layout for A, C, D, E, F; B is placed by hand afterwards
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), layout="constrained")
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=4 / 72, wspace=0.02, hspace=0.02)
    gs = fig.add_gridspec(4, 1, height_ratios=ROW_HEIGHTS)

    row1 = gs[0].subgridspec(1, 2, width_ratios=[1.0, 1.7], wspace=0.04)   # A | B
    ax_A, ax_B = fig.add_subplot(row1[0]), fig.add_subplot(row1[1])
    ax_B.axis("off")
    ax_C, ax_D = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    row4 = gs[3].subgridspec(1, 3, width_ratios=[0.75, 1.9, 0.5], wspace=0.04)   # E | F | legends
    ax_E, ax_F, ax_leg = (fig.add_subplot(row4[i]) for i in range(3))
    ax_leg.axis("off")

    panel_A(ax_A, protein)
    cell_cols = panel_C(ax_C, d)
    panel_D(ax_D, protein)
    panel_E(ax_E, d)
    panel_F(ax_F, ax_leg, d.rna_t, yticks=[0, 0.5, 1, 1.5])

    # draw once to measure C's tick labels for the brackets, again so the layout reserves their space, then freeze it
    fig.canvas.draw()
    add_group_brackets(ax_C, SCCODA_GROUPS, cell_cols, fig.canvas.get_renderer())
    fig.canvas.draw()
    fig.set_layout_engine("none")
    fig.canvas.draw()

    # align the left edges (tick labels included) of A, C, D, E; the right edges stay
    renderer, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    left_axes = [ax_A, ax_C, ax_D, ax_E]
    tight_x0 = [a.get_tightbbox(renderer).transformed(inv).x0 for a in left_axes]
    target = max(min(tight_x0), 0.002)
    for a, x0 in zip(left_axes, tight_x0):
        pos, shift = a.get_position(), target - x0
        a.set_position([pos.x0 + shift, pos.y0, pos.width - shift, pos.height])

    ax_B_axes = panel_B(fig, ax_B, protein, d)
    fig.canvas.draw()

    def panel_labels(row):
        """row = [(letter, [axes]), ...]; the letters share one height, at the top of the highest element."""
        bbs = {L: [a.get_tightbbox(renderer).transformed(inv) for a in axs] for L, axs in row}
        y = max(b.y1 for v in bbs.values() for b in v)
        for L, bb in bbs.items():
            fig.text(max(min(b.x0 for b in bb), 0.002), y, L, fontsize=FS, fontweight="bold", ha="left", va="top")

    panel_labels([("A", [ax_A]), ("B", list(ax_B_axes))])
    panel_labels([("C", [ax_C])])
    panel_labels([("D", [ax_D])])
    panel_labels([("E", [ax_E]), ("F", [ax_F])])

    if save:   # no bbox_inches="tight" so the page size stays exact
        for ext in ("png", "pdf"):
            fig.savefig(f"{fig_dir}/{fig_name}.{ext}", dpi=600)
    plt.show()
    plt.rcdefaults()
    return fig
