import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy.sparse import issparse
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.utils import sparsefuncs
from mudata import MuData
import knn_normalization as knn


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
