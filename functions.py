import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import muon
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import scperturb
import os
from scipy import stats
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.sparse import issparse
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from warnings import warn
from scipy.optimize import curve_fit
from typing import Optional, Union, Literal
from numbers import Integral
from anndata import AnnData
from mudata import MuData
import knn_normalization as knn
import statsmodels.api as sm
from sklearn.utils import sparsefuncs

def neighbors_umap(data, use_PCA=True, metric="euclidean", n_neighbors=15):
    
    if use_PCA:
        sc.pp.neighbors(data, metric=metric, n_neighbors=n_neighbors)
    else:
        sc.pp.neighbors(data, use_rep="X", metric=metric, n_neighbors=n_neighbors)
    sc.tl.umap(data)

def pca_neighbors_umap(data, n_comps=None, n_neighbors=15):
    
    sc.pp.pca(data, n_comps=n_comps) # I think the default is to use all proteins for PCA. Check the default. Think about maybe using only highly variable proteins for PCA.
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
    
    sc.pp.pca(data, n_comps=n_comps) # I think the default is to use all proteins for PCA. Check the default. Think about maybe using only highly variable proteins for PCA.
    sc.pp.neighbors(data, n_neighbors=n_neighbors)
    sc.tl.umap(data)
    sc.tl.leiden(data, flavor="igraph", n_iterations=2, directed=False)

def preprocess_rna(data):
    """
    Usual RNA preprocessing: gene and cell filtering, library-size normalization and log-scaling, 
    PCA, UMAP.
    """
    sc.pp.filter_genes(data, min_counts=1)
    sc.pp.filter_cells(data, min_genes=200)
    sc.pp.normalize_total(data)
    sc.pp.log1p(data)
    pca_neighbors_umap(data) # This chunk takes quite a lot of time.


def sum_total_counts(data, obs_names=["total_counts", "log_total_counts"], log1p=False, layer=None):
    """
    Function to sum total per-cell counts.
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
    Function to sum total per-cell counts. This function is used when I have already taken the logarithm of the data.
    Therefore, I take the exponential before summing counts.
    """

    if expm1:
        if layer is None:
            data.layers["exponential"] = np.expm1(data.X)
        else:
            data.layers["exponential"] = np.expm1(data.layers[layer])
    else:
        if layer is None:
            data.layers["exponential"] = np.exp(data.X)
        else:
            data.layers["exponential"] = np.exp(data.layers[layer])
    total_counts = data.layers["exponential"].sum(axis=1)
    data.obs[obs_names[0]] = total_counts
    log_total_counts = np.log10(total_counts)
    data.obs[obs_names[1]] = log_total_counts
    
    del data.layers["exponential"]  

def create_protein_anndata(data, protein_data_name="protein_knn_normalized", logged_or_raw="raw", use_PCA=True):
    """
    Function to create an Anndata object from the protein (ADT) data in a CITE-seq dataset. Expects the name of the adata.obsm object that contains the ADT data.
    """

    protein_anndata = sc.AnnData(data.obsm[protein_data_name].copy(), dtype=float)
    protein_anndata.obs = data.obs
    if use_PCA:
        sc.pp.pca(protein_anndata)
    neighbors_umap_leiden(protein_anndata, use_PCA=use_PCA)
    
    # Calculating total counts per cell.
    if logged_or_raw == "logged":
        sum_total_counts_logged(protein_anndata)
    elif logged_or_raw == "raw":
        sum_total_counts(protein_anndata)
    
    # Doing CLR (Centered log-ratio) transformation and storing it as a layer. You can do CLR normalization across cells (axis=1), or across antibodies (axis=0).
    protein_anndata.layers["clr_across_cells"] = muon.prot.pp.clr(protein_anndata, inplace=False, axis=1).X
    protein_anndata.layers["clr_across_ab"] = muon.prot.pp.clr(protein_anndata, inplace=False, axis=0).X
    
    return protein_anndata


def create_protein_anndata_from_layer(data, layer_name="clr_across_cells", logged_or_raw="raw", use_PCA=True):
    """
    Function to create an Anndata object from the CLR-normalized data. Expects an Anndata object where one of the layers
    is the CLR-normalized protein data.
    """
    
    # Copy the clr-normalized layer data onto the main data.
    new_anndata = data.copy() # It's very important that you use .copy(). If you just said clr_normalized = data, then clr_normalized would be just a view of data, and all changes applied to clr_normalized would be also applied to data. That is, you wouldn't have two different object, just one object with two different names, which is not what we want.
    new_anndata.X = new_anndata.layers[layer_name]
    
    # Delete the layer data, because it's already in the main data.
    delete_clr_layers(new_anndata)
    
    # Calculating total counts per cell.
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
    Function to calculate the maximum ARI, choosing from different resolutions, where the number of clusters produced by the resolution has to be in a bounded region of +-20% of the number of cell types in the cell type annotations.
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
                    ari_list.append([ari, res_number,  clusters])
                ari_df_clustering = pd.DataFrame(ari_list, columns = ["ARI", "Resolution", "n_clusters"])
                ari_bounded = ari_df_clustering[ari_df_clustering["n_clusters"].between(nclusters_annotations - nclusters20percent, nclusters_annotations + nclusters20percent, inclusive="both")] # Create a dataframe of ARI values within the +-20% bounded region of number of cell types.
                if ari_bounded.shape[0] == 0: # If there are no clustering resolutions within the +-20% bounded region of number of cell types, I just take the clustering resolution that produces a number of clusters as similar as possible to the number of cell types.
                    diff_to_nclusters_annotations = abs(ari_df_clustering["n_clusters"] - nclusters_annotations)
                    closest_resolutions_to_nclusters_annotations = ari_df_clustering[diff_to_nclusters_annotations == diff_to_nclusters_annotations.min()]
                    best_row = closest_resolutions_to_nclusters_annotations[closest_resolutions_to_nclusters_annotations["ARI"] == closest_resolutions_to_nclusters_annotations["ARI"].max()]
                    ari_max, ari_max_resolution, ari_max_clusters = float(best_row["ARI"].iloc[0]), float(best_row["Resolution"].iloc[0]), int(best_row["n_clusters"].iloc[0])
                else:
                    ari_max = float(ari_bounded["ARI"].max())
                    ari_max_resolution = float(ari_bounded[ari_bounded["ARI"] == ari_max]["Resolution"].iloc[0])
                    ari_max_clusters = int(ari_bounded[ari_bounded["ARI"] == ari_max]["n_clusters"].iloc[0])
                ari_dict[dataset] = [round(ari_max, 3), ari_max_resolution, ari_max_clusters]
        ari_df = pd.DataFrame(ari_dict, index = ["ARI", "Resolution", "n_clusters"]).T
        ari_df.round(3).to_csv(f"{save_directory}/ari_scores_{cell_type_annotation}.csv", index=True)
        ari_df_simplified = average_across_samples(ari_df, f"{save_directory}/ari_scores_simplified_{cell_type_annotation}.csv",
                                                             sample_word_number = sample_word_number)
        ari_df_simplified = ari_df_simplified.reset_index().rename(columns={"index": "assay"})
        boxplot_comparing_normalizations(ari_df, savefilename=f"{save_directory}/ARI_{cell_type_annotation}_by_method", profi=profi, metric="ARI", 
                                         selected_dataset_and_methods=selected_dataset_and_methods, sample_word_number=sample_word_number, 
                                         methods_palette=methods_palette, figsize=figsize, rcparams=rcparams, orientation=orientation, xticks=xticks)

def knn_classifier_varying_k(
    data_samples,
    k_values=(20, 40, 60, 80),
    classification="Smooth_cell_type",
    sample_word_number=1
):
    """
    Sweep k for your KNN classifier across all samples and assay types.
    Robust (Fix A): builds dict rows so DataFrame columns always match.
    Reuses knn_classifier_normalization() / knn_classifier().
    """

    keys = list(data_samples.keys())

    # Sample name = first `sample_word_number` tokens
    samples = sorted(set("_".join(k.split("_")[:sample_word_number]) for k in keys))

    # Assay/type = everything after the sample prefix
    assays = sorted(set("_".join(k.split("_")[sample_word_number:]) for k in keys))

    # Map (sample, assay) -> one key in data_samples (pick first deterministically)
    sa2key = {}
    for k in sorted(keys):
        s = "_".join(k.split("_")[:sample_word_number])
        a = "_".join(k.split("_")[sample_word_number:])
        sa2key.setdefault((s, a), k)

    rows = []

    for sample in samples:
        # Precompute dataset keys for this sample in the order of `assays`
        data_types = [sa2key.get((sample, a)) for a in assays]

        # If sample has no data at all, skip
        if all(dt is None for dt in data_types):
            continue

        for k in k_values:
            # Build a dict row (Fix A)
            row = {"Sample": sample, "n_cells": np.nan, "k": int(k)}

            # Compute accuracies assay-by-assay, filling NaN if missing
            for assay, dataset_key in zip(assays, data_types):
                if dataset_key is None:
                    row[assay] = np.nan
                    continue

                adata = data_samples[dataset_key]

                # Keep behavior consistent with your knn_classifier()
                if adata.n_obs < k:
                    row[assay] = np.nan
                    continue

                acc = knn_classifier(adata, k_neighbor_number=int(k), classification=classification)
                # knn_classifier returns "not_enough_cells" or float; we store NaN for the string
                row[assay] = np.nan if acc == "not_enough_cells" else acc

                # Fill shared metadata
                row["n_cells"] = adata.n_obs

            rows.append(row)

    knn_accuracy = pd.DataFrame(rows)

    # Ensure stable column order
    desired_cols = ["Sample", "n_cells", "k"] + assays
    for c in desired_cols:
        if c not in knn_accuracy.columns:
            knn_accuracy[c] = np.nan
    knn_accuracy = knn_accuracy[desired_cols]

    # simplified = mean over k per Sample x assay
    simplified = (
        knn_accuracy
        .melt(id_vars=["Sample", "n_cells", "k"], var_name="assay", value_name="knn_classifier_accuracy")
        .dropna(subset=["knn_classifier_accuracy"])
        .groupby(["Sample", "assay"], as_index=False)
        .agg(n_cells=("n_cells", "first"),
             knn_classifier_accuracy=("knn_classifier_accuracy", "mean"))
    )

    return knn_accuracy, simplified

def knn_classifier(data, k_neighbor_number=15, classification="Smooth_cell_type"):
    """Function to calculate the classification accuracy of an ADT dataset based on the neighbors of each cell.
    So we classify each cells based on its neighbors. Conveys very similar information to UMAP, but numerically."""
    
    if data.n_obs < k_neighbor_number:   
        return "not_enough_cells"
    else:
        knn_classifier = KNeighborsClassifier(k_neighbor_number)
        knn_classifier.fit(data.X, data.obs[classification]) # Fits each cell's ADT counts to that cell's classification.
        accuracy = knn_classifier.score(data.X, data.obs[classification]) # It looks weird that I score in the training data, but that's what I want: this cell's classification based on its neighbors.
        # print(accuracy)
        return round(accuracy, 3) # Round the accuracy to 3 decimals.


def knn_classifier_varying_k_onesample(data, k_values=[1, 5, 10, 15, 20, 30, 40, 80], classification="Smooth_cell_type"):
    """Function to make a sweep of k-values of the KNN classifier."""
    
    knn_accuracies = []
    
    # Gather the different data types (KNN-normalized, CLR-normalized, etc.) for each sample.

    sample_data_types = []
    for dataset_name in data.keys():
        if ("RNA" not in dataset_name or "KNN" in dataset_name): # Here I get the data types that contain the sample of interest, and excluding the RNA data because it takes a lot of time.
            sample_data_types.append(dataset_name)
                
    # Making the sweep of K-values of the KNN classifier.        
        for k_value in k_values: 
            print(k_value)
            knn_classifier_accuracy = knn_classifier_normalization_onesample(sample_data_types, data, k_neighbor_number=k_value, classification=classification)
            knn_accuracies.append(knn_classifier_accuracy)
    
    # Create a Dataframe with all clustering coefficients.
    knn_accuracies = pd.DataFrame(knn_accuracies, columns = ["k"] + sample_data_types)  # NAME COLUMNS
    # knn_accuracies.sort_values(["Sample", "k"], inplace=True)
            
    return knn_accuracies

def knn_classifier_normalization_onesample(data_types, data, k_neighbor_number=15, classification="Smooth_cell_type"):
    """Function to calculate the classification accuracy of an ADT dataset based on the neighbors of each cell.
    So we classify each cells based on its neighbors. Conveys very similar information to UMAP, but numerically. The function
    calculates the classfication accuracy for different data types: raw protein data, knn-normalized data, CLR-normalized data, etc."""
    
    for i,data_type in enumerate(data_types):
        if i == 0:           
            data_accuracy_stats = [k_neighbor_number]
        knn_classification_accuracy = knn_classifier(data[data_type], k_neighbor_number, classification)
        data_accuracy_stats.append(knn_classification_accuracy)
        
    return data_accuracy_stats

def plot_KNN_classifier_accuracies(knn_accuracies, simplified_knn_accuracies, save_directory, cell_type_annotation, legend_outside_plot=False):
    """
    Function to create a barplot from KNN classifier accuracies. Plots the 
    average KNN classifier accuracy for each sample.
    """
    
    # knn_accuracies = knn_accuracies[knn_accuracies["n_cells"]>450] # Remove untrustworthy samples, with less than 450 cells.
    knn_accuracies = knn_accuracies.drop(columns="k") # We don't need the k number because we are going to take the mean across k values.
    knn_accuracies = knn_accuracies.groupby("Sample").mean(numeric_only=False).reset_index()
    knn_accuracies_long = pd.melt(knn_accuracies, id_vars=["Sample", "n_cells"],
                                  var_name="assay", value_name="knn_classifier_accuracy") # Change the dataframe to a long format in order to plot it.
    # Plot KNN accuracies by sample
    sns.barplot(data = knn_accuracies_long, x="Sample", y="knn_classifier_accuracy", hue="assay")
    if legend_outside_plot:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
    else:
        plt.legend(loc="lower right")
    plt.title(f"KNN classifier accuracy")
    plt.tight_layout()
    savefilename = save_directory + f"/KNN_classifier_accuracy_{cell_type_annotation}.pdf"
    plt.savefig(savefilename)
    plt.close("all")
    
    # Plot simplified KNN accuracies: average across k values and across samples.
    
    plot_simplified_table(df=simplified_knn_accuracies, title="KNN classifier accuracy", 
                          savefilename=save_directory + f"/simplified_KNN_classifier_accuracy_{cell_type_annotation}.pdf")
    
    # simp_barplot = sns.barplot(data=simplified_knn_accuracies, x="assay", y="KNN_accuracy", hue="assay")
    # plt.title(f"KNN classifier accuracy")
    # savefilename = save_directory + f"/simplified_KNN_classifier_accuracy_{cell_type_annotation}.pdf"
    # simp_barplot.set(xticklabels=[])
    # simp_barplot.tick_params(bottom=False)
    # plt.savefig(savefilename)
    # plt.close("all")

def calculate_knn_classifier_accuracy(data_samples, cell_annotation_levels, k_values_knn_classifier, working_directory, profi, 
                                      selected_dataset_and_methods, sample_word_number, methods_palette, rcparams, 
                                      figsize=(2.12, 1.60), orientation="vertical", xticks=None):
    for cell_type_annotation in cell_annotation_levels:
        knn_accuracies, simplified = knn_classifier_varying_k(data_samples, k_values = k_values_knn_classifier, classification=cell_type_annotation,
                                                                      sample_word_number = sample_word_number)
        simplified.to_csv(f"{working_directory}/figures/simplified_KNN_classifier_{cell_type_annotation}.csv", index=False)
        knn_accuracies.to_csv(f"{working_directory}/figures/KNN_classifier_{cell_type_annotation}.csv", index=False)
        plot_KNN_classifier_accuracies(knn_accuracies, simplified, save_directory = f"{working_directory}/figures", 
                                                 cell_type_annotation = cell_type_annotation, legend_outside_plot=True)
        
        # knn_accuracies = knn_accuracies[knn_accuracies["n_cells"]>450] # Remove untrustworthy samples, with less than 450 cells.
        knn_accuracies = knn_accuracies.drop(columns="k") # We don't need the k number because we are going to take the mean across k values.
        knn_accuracies_long = pd.melt(knn_accuracies, id_vars=["Sample", "n_cells"],
                                      var_name="assay", value_name="knn_classifier_accuracy") # Change the dataframe to a long format in order to plot it.
        knn_accuracies_long.index=knn_accuracies_long["Sample"] + "_" + knn_accuracies_long["assay"]
        boxplot_comparing_normalizations(knn_accuracies_long, savefilename=f"{working_directory}/figures/KNN_classifier_accuracy_bloxplot_{cell_type_annotation}_by_method", 
                                         profi=profi, metric="knn_classifier_accuracy", selected_dataset_and_methods=selected_dataset_and_methods, 
                                         sample_word_number=sample_word_number, methods_palette=methods_palette, figsize=figsize, 
                                         rcparams=rcparams, orientation=orientation, xticks=xticks)


def plot_histogram_total_counts(data_sample, plot_title, save_filename, cell_type_annotations=None, log_transform=False, xlim=None):
    """
    Plots a histogram of total counts per cell in the data of a patient sample.
    """
    
    sum_total_counts(data_sample)
    if log_transform:
        if cell_type_annotations is None:
            sns.histplot(data_sample.obs, x="log_total_counts")
        else:
            sns.histplot(data_sample.obs, x="log_total_counts", hue=cell_type_annotations)
        plt.title(plot_title)
        plt.xlabel("Log total counts in the cell")
        plt.ylabel("Number of cells")
        plt.savefig(save_filename)
    else:
        if cell_type_annotations is None:
            sns.histplot(data_sample.obs, x="total_counts")
        else:
            sns.histplot(data_sample.obs, x="total_counts", hue=cell_type_annotations)
        if xlim:
            plt.xlim(xlim)
        plt.title(plot_title)
        plt.xlabel("Total counts in the cell")
        plt.ylabel("Number of cells")
        plt.savefig(save_filename)
    
    
    plt.close("all")


def histogram_total_protein_counts_by_cell_type_data_samples(data_samples, cell_type_annotations, log_transform=False):
    """
    
    data_samples: A dictionary of AnnData objects.
    cell_type_annotations: something like celltype.l1
    """
    for dataset in data_samples.keys():
        if log_transform:
            sns.histplot(data_samples[dataset].obs, x="total_counts", hue=cell_type_annotations)
            plt.xlabel("Total counts in the cell")
            plt.ylabel("Number of cells")
            plt.title(f"Histogram of total {dataset} counts by cell type")
            plt.savefig(f"figures/Histogram_total_{dataset}_by_cell_type.pdf")
            
        else:
            sns.histplot(data_samples[dataset].obs, x="log_total_counts", hue=cell_type_annotations)
            plt.xlabel("Log total counts in the cell")
            plt.ylabel("Number of cells")
            plt.title(f"Histogram of total {dataset} counts by cell type")
            plt.savefig(f"figures/Histogram_total_{dataset}_by_cell_type.pdf")
            
        plt.close("all")


def take_geometric_mean(vector, log1p=False):
    
    if log1p:
        geometric_mean = np.exp(np.log1p(vector).mean())
    else:
        geometric_mean = np.exp(np.log(vector).mean())
        
    return geometric_mean


def clr_transform_vector(vector, log1p=False):
    """
    Calculates CLR transformation of a vector. Attention, here I use log instead of log1p, whereas the standard CLR transformation uses log1p to deal with zeros.
    """
    vector = np.array(vector)
    geometric_mean = take_geometric_mean(vector, log1p=log1p)

    if log1p:  
        clr_transformed_vector = np.log1p(vector / geometric_mean)
    else:
        clr_transformed_vector = np.log(vector / geometric_mean)
    return clr_transformed_vector


def clr_transform_matrix(matrix, axis=1, log1p=True, log_transform=True, return_geom_means=False,
                         flavor="seurat"):
    """
    Homemade function to calculate CLR.
    
    When using log1p, the geometric means are not strictly the geometric mean because a +1 
    pseudocount has been added. But this is what CLR normalization uses, so it's what is important
    for me. 
    """

    if flavor=="seurat":
        if log1p:
            geom_means = np.exp(np.log1p(matrix).sum(axis=axis, keepdims=True) / matrix.shape[axis])
            clr_transformed_matrix = matrix / geom_means
        else:
            geom_means = np.exp(np.log(matrix).sum(axis=axis, keepdims=True) / matrix.shape[axis])
            clr_transformed_matrix = matrix / geom_means
            
        if log_transform:
            if log1p:
                clr_transformed_matrix = np.log1p(clr_transformed_matrix)
            else:
                clr_transformed_matrix = np.log(clr_transformed_matrix)

    elif flavor=="stoeckius":
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


def obtain_common_columns_and_rows(list_of_heatmaps, cell_type_annotation):
    """
    Function to obtain the common columns and rows from a list of E-distance heatmaps.
    """
    common_columns = set(list_of_heatmaps[0].columns)

    for heatmap in list_of_heatmaps[1:]:
        common_columns = common_columns.intersection(set(heatmap.columns))
    common_columns = list(common_columns)
    common_rows = [column for column in common_columns if column!=cell_type_annotation] # Remove the cell type annotation because it's a column but not a row.
    
    return common_columns, common_rows

def drop_outlier_cell_types(list_of_heatmaps, cell_type_annotation, common_columns, common_rows):
    """
    Function to drop outlier cell types (cell types not contained in other samples).
    """
    new_heatmaps = []
    for heatmap in list_of_heatmaps:
        columns_to_drop = [column for column in heatmap.columns if column not in common_columns]
        heatmap = heatmap.drop(columns=columns_to_drop) # Drop outlier columns.
        heatmap = heatmap[heatmap[cell_type_annotation].isin(common_rows)] # Drop outlier rows.
        new_heatmaps.append(heatmap)
    
    return new_heatmaps


def filter_small_cell_types(data, cell_type_annotation, n_cells_threshold=3):
    """
    Function to remove cell types with very few cells.
    """
    cell_type_counts = data.obs[cell_type_annotation].value_counts()
    cell_types_to_remove = cell_type_counts[cell_type_counts < 3].index
    data_filtered = data[~data.obs[cell_type_annotation].isin(cell_types_to_remove)]
    
    return data_filtered.copy()

def analyze_KNN_normalized_protein(protein_knn_normalized, total_size_factors, size_factor_history, use_PCA=True):
    """
    Function to do a basic analysis of KNN normalized protein, like running neighbors and UMAP, or adding
    the total size factor to .obs. 
    """
    
    protein_knn_normalized.obs["total_size_factor"] = total_size_factors
    protein_knn_normalized.obs["inverse_total_size_factor"] = 1/protein_knn_normalized.obs.total_size_factor
    protein_knn_normalized.obsm["size_factor_history"] = size_factor_history
    if use_PCA:
        sc.pp.pca(protein_knn_normalized)
    neighbors_umap_leiden(protein_knn_normalized, use_PCA=use_PCA)

def apply_normalization_methods(protein_data, rna_data=None, use_PCA_for_protein=True, verbose=True,
                                log_transform=True, pseudocount_value=5, n_neighbors=None,
                                scale_protein=True, preprocess_rna=True):
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

    # Calculating PCA(if necessary), neighbors, UMAPs.
    if verbose:
        print("PCA_neighbors_UMAP")
    if rna_data is not None:
        pca_neighbors_umap_leiden(rna_data)
    
    if use_PCA_for_protein:
        sc.pp.pca(protein_data)
    neighbors_umap_leiden(protein_data, use_PCA=use_PCA_for_protein)

    # Save AnnData object from RNA and protein data.
    if verbose:
        print("Storing Anndata objects")
    if rna_data is not None:
        normalized_data["RNA"] = rna_data.copy()
    if scale_protein:
        protein_scaled = protein_data.copy()
        sc.pp.scale(protein_scaled)
        normalized_data["protein_raw"] = protein_scaled.copy()

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
    # sc.pp.normalize_total(library_normalization)
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
    neighbors_umap_leiden(CLR_normalization, use_PCA = use_PCA_for_protein)

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
    neighbors_umap_leiden(CLR_normalization, use_PCA = use_PCA_for_protein)

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
    neighbors_umap_leiden(CLR_normalization, use_PCA = use_PCA_for_protein)

    normalized_data["CLR_normalization_across_ab"] = CLR_normalization.copy()
    normalized_data["CLR_normalization_across_ab"].var["geom_mean"] = geom_means.flatten()

    # KNN normalization with RNA neighbors
    if rna_data is not None:
        if verbose:
            print("KNN normalization with RNA neighbors")

        if preprocess_rna:
            mudata = MuData({"prot": protein_data, "rna": rna_data})
            mudata_knn_normalized = knn.knn_normalize(mudata, calculate_neighbors_from="rna",
                                                    log_transform=log_transform, n_neighbors=n_neighbors, pseudocount=pseudocount_value,
                                                    inplace=False, save_size_factors=True, verbose=verbose)
            protein_knn_normalized = mudata_knn_normalized["prot"]
        else:
            protein_knn_normalized = protein_data.copy()
            rna = rna_data.copy()
            sc.pp.pca(rna)
            neigh = n_neighbors if n_neighbors is not None else max(15, min(round(rna.n_obs / 20), 300))
            sc.pp.neighbors(rna, n_neighbors=neigh)
            protein_knn_normalized.obsp["connectivities"] = rna.obsp["connectivities"]
            knn.knn_normalize(protein_knn_normalized, calculate_neighbors_from="use_existing_neighbors",
                              log_transform=log_transform, n_neighbors=n_neighbors,
                              pseudocount=pseudocount_value, save_size_factors=True, inplace=True, verbose=verbose)

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
                         log_transform=True, pseudocount_value=5, n_neighbors=None, scale_protein=True, 
                         preprocess_rna=True):
    """
    Applies different normalization procedures to each samples separately. Returns a dictionary called 
    "data_samples" which contains all the normalized data.
    
    sample_list: a list with the names of the samples for which the normalization procedure should be applied.
    sample_column_name: name of the column in data.obs that indicates the sample to which each cell belongs.
    """

    data_samples = {}
    for sample in sample_list: # For each sample.

        print(sample)
        # Select the data from the sample.
        if rna_data is not None:
            sample_rna = rna_data[rna_data.obs[sample_column_name] == sample].copy() 
        else:
            sample_rna = None
        sample_protein = protein_data[protein_data.obs[sample_column_name] == sample].copy() 
        
        # Apply the different normalization methods to the protein data.
        normalized_data = apply_normalization_methods(protein_data=sample_protein, rna_data=sample_rna, use_PCA_for_protein=use_PCA_for_protein,
                                                      verbose=verbose, log_transform=log_transform, preprocess_rna=preprocess_rna,
                                                      pseudocount_value=pseudocount_value, n_neighbors=n_neighbors, scale_protein=scale_protein)
        # Add the sample name in the resulting dictionary.
        normalized_data = {f"{sample}_{key}": value for key,value in normalized_data.items()}
        
        # Update the dictionary that contains the normalized data with all the samples, by including the dictionary from this one sample.
        data_samples.update(normalized_data)
        
    return data_samples

def plot_scatterplot_with_correlation(data, x, y, title, filename):
    """
    Function to plot a scatterplot with Pearson correlation. 
    
    x: Variable to plot on the x axis of the scatterplot.
    y: Variable to plot on the y axis of the scatterplot.
    """
    
    plot = sns.jointplot(data=data, x=x, y=y, kind='reg')
    r,p = stats.pearsonr(x=data[x], y=data[y])
    legend, = plot.ax_joint.plot([], [], linestyle="", alpha=0)
    plot.ax_joint.legend([legend],[f'Pearsonr={r:.2f}'])
    plt.suptitle(title)
    plt.savefig(f"figures/{filename}")
    plt.show()


def remove_PCA(data):
    
    del data.obsm["X_pca"]
    del data.varm["PCs"]
    del data.uns["pca"]


def plot_marker_genes(data_samples, marker_genes, samples, save_directory, cell_types_palette, cell_type_annotation="Smooth_cell_type"):
    """
    A function to plot the expression of marker genes across the different cell types and with different normalizations.
    """
    for sample in samples:
        for marker_gene in marker_genes:
            fig,axs = plt.subplots(2,2, figsize = (10,10))
            sc.pl.violin(data_samples[f"{sample}_protein_uncorrected"], keys=marker_gene, groupby=cell_type_annotation, rotation=90, ax=axs[0,0], show=False, use_raw=False, palette=cell_types_palette)
            axs[0,0].set_title(f"Uncorrected")

            sc.pl.violin(data_samples[f"{sample}_KNN_RNA_neighbors"], keys=marker_gene, groupby=cell_type_annotation, rotation=90, ax=axs[0,1], show=False, use_raw=False, palette=cell_types_palette)
            axs[0,1].set_title(f"KNN normalization, RNA neighbors")

            sc.pl.violin(data_samples[f"{sample}_CLR_normalization"], keys=marker_gene, groupby=cell_type_annotation, rotation=90, ax=axs[1,0], show=False, use_raw=False, palette=cell_types_palette)
            axs[1,0].set_title(f"CLR normalization")

            sc.pl.violin(data_samples[f"{sample}_KNN_RNA_neighbors_cosine"], keys=marker_gene, groupby=cell_type_annotation, rotation=90, ax=axs[1,1], show=False, use_raw=False, palette=cell_types_palette)
            axs[1,1].set_title(f"KNN normalization, protein neighbors")

            savefilename = f"{save_directory}/violin_plot_SNR_{sample}_{marker_gene}_allcells.pdf" 
            fig.tight_layout()
            fig.savefig(savefilename)
            plt.close("all")


def signal_to_noise_ratio(data_samples, marker_gene, dataset, median=True, sparse=False):
    """
    Calculate the signal to noise ratio of markers in cell types. This SNR score will be the difference between the expression
    of the marker in the target cells and the expression of the marker in other cells, divided by the standard deviation.
    """
    if sparse:
        marker_gene_target_cell = np.array(data_samples[dataset][data_samples[dataset].obs["group_cell_type"] == "Target cells"][:, marker_gene].X.toarray())
        marker_gene_other_cells = np.array(data_samples[dataset][data_samples[dataset].obs["group_cell_type"] == "Other cells"][:, marker_gene].X.toarray())
    else:
        marker_gene_target_cell = np.array(data_samples[dataset][data_samples[dataset].obs["group_cell_type"] == "Target cells"][:, marker_gene].X)
        marker_gene_other_cells = np.array(data_samples[dataset][data_samples[dataset].obs["group_cell_type"] == "Other cells"][:, marker_gene].X)
    
    if median:
        score = (np.median(marker_gene_target_cell) - np.median(marker_gene_other_cells)) / (np.std(marker_gene_target_cell) + np.std(marker_gene_other_cells))
    else:
        score = (np.mean(marker_gene_target_cell) - np.mean(marker_gene_other_cells)) / (np.std(marker_gene_target_cell) + np.std(marker_gene_other_cells))
        
    return score

def snr_plot_scores(data_samples, marker_genes, target_cell_types_list, excluded_cell_types_list, samples, save_directory, cell_type_annotation="Smooth_cell_type", median=True, sparse=False):
    """
    Calculate the Signal-to-noise ratio for each marker gene, and plot the expression of the marker gene in the target cell type vs other cell types.
    
    excluded_cell_types_list: cell types that are not target cell types, but also not fully negative cell types. They might have a subpopulation that expressed the marker.
    We exclude them from the signal-to-noise ratio calculation.
    """

    for i in range(len(marker_genes)):
        marker_gene = marker_genes[i]
        target_cell_types = target_cell_types_list[i]
        excluded_cell_types = excluded_cell_types_list[i]
        for dataset in data_samples.keys():
            data_samples[dataset].obs["group_cell_type"] = data_samples[dataset].obs[cell_type_annotation].apply(lambda x: "Target cells" if x in target_cell_types else ("Excluded cells" if x in excluded_cell_types else "Other cells"))
            data_samples[dataset].obs["group_cell_type"] = data_samples[dataset].obs["group_cell_type"].str.strip() # Otherwise blank spaces cause an error when plotting.
        for sample in samples:
            fig,axs = plt.subplots(2,2, figsize=(10,10))
            score = signal_to_noise_ratio(data_samples, marker_gene, dataset=f"{sample}_protein_uncorrected", median=median, sparse=sparse)
            filtered_logged = data_samples[f"{sample}_protein_uncorrected"][data_samples[f"{sample}_protein_uncorrected"].obs["group_cell_type"] != "Excluded cells"].copy()
            sc.pl.violin(filtered_logged, keys=marker_gene, groupby="group_cell_type", rotation=90, ax=axs[0,0], show=False, use_raw=False)
            axs[0,0].set_title(f"{sample} Logged protein, score={str(round(score,2))}")
            
            # score = signal_to_noise_ratio(data_samples, marker_gene, dataset=f"{sample}_protein_raw", median=median, sparse=sparse)
            # sc.pl.violin(data_samples[f"{sample}_protein_raw"], keys=marker_gene, groupby="group_cell_type", rotation=90, ax=axs[0,0], log=True, show=False, use_raw=False)
            # axs[0,0].set_title(f"{sample} raw protein, score={round(score,2)}")
            
            score = signal_to_noise_ratio(data_samples, marker_gene, dataset=f"{sample}_KNN_protein_neighbors", median=median, sparse=sparse)
            filtered_knn_cosine = data_samples[f"{sample}_KNN_protein_neighbors"][data_samples[f"{sample}_KNN_protein_neighbors"].obs["group_cell_type"] != "Excluded cells"].copy()
            sc.pl.violin(filtered_knn_cosine, keys=marker_gene, groupby="group_cell_type", rotation=90, ax=axs[0,1], show=False, use_raw=False)
            axs[0,1].set_title(f"{sample} KNN-normalized cosine byAb, score={str(round(score,2))}")
            
            score = signal_to_noise_ratio(data_samples, marker_gene, dataset=f"{sample}_CLR_normalization", median=median, sparse=sparse)
            filtered_clr=data_samples[f"{sample}_CLR_normalization"][data_samples[f"{sample}_CLR_normalization"].obs["group_cell_type"] != "Excluded cells"].copy()
            sc.pl.violin(filtered_clr, keys=marker_gene, groupby="group_cell_type", rotation=90, ax=axs[1,0], show=False, use_raw=False)
            axs[1,0].set_title(f"{sample} CLR-normalized protein, score={str(round(score,2))}")
            
            score = signal_to_noise_ratio(data_samples, marker_gene, dataset=f"{sample}_KNN_RNA_neighbors", median=median, sparse=sparse)
            filtered_knn = data_samples[f"{sample}_KNN_RNA_neighbors"][data_samples[f"{sample}_KNN_RNA_neighbors"].obs["group_cell_type"] != "Excluded cells"].copy()
            sc.pl.violin(filtered_knn, keys=marker_gene, groupby="group_cell_type", rotation=90, ax=axs[1,1], show=False, use_raw=False)
            axs[1,1].set_title(f"{sample} KNN-normalized byAb, score={str(round(score,2))}")
            
            savefilename = f"{save_directory}/violin_plot_SNR_{sample}_{marker_gene}.pdf"
            fig.tight_layout()
            fig.savefig(savefilename)
            plt.close("all")
            
def snr_plot_scores_targeted_analysis(data_samples, marker_genes, target_cell_types_list, excluded_cell_types_list, samples, save_directory, cell_type_annotation="celltype_nk_t_cells", median=False, sparse=True):
    """
    Calculate the Signal-to-noise ratio for each marker gene, and plot the expression of the marker gene in the target cell type vs other cell types.
    
    excluded_cell_types_list: cell types that are not target cell types, but also not fully negative cell types. They might have a subpopulation that expressed the marker.
    We exclude them from the signal-to-noise ratio calculation.
    """

    for i in range(len(marker_genes)):
        marker_gene = marker_genes[i]
        target_cell_types = target_cell_types_list[i]
        excluded_cell_types = excluded_cell_types_list[i]
        for dataset in data_samples.keys():
            data_samples[dataset].obs["group_cell_type"] = data_samples[dataset].obs[cell_type_annotation].apply(lambda x: "Target cells" if x in target_cell_types else ("Excluded cells" if x in excluded_cell_types else "Other cells"))
            data_samples[dataset].obs["group_cell_type"] = data_samples[dataset].obs["group_cell_type"].str.strip() # Otherwise blank spaces cause an error when plotting.
        for sample in samples:
            fig,axs = plt.subplots(1,3, figsize=(10,10))
            score = signal_to_noise_ratio(data_samples, marker_gene, dataset=f"{sample}_not_targeted", median=median, sparse=sparse)
            filtered_not_targeted = data_samples[f"{sample}_not_targeted"][data_samples[f"{sample}_not_targeted"].obs["group_cell_type"] != "Excluded cells"].copy()
            sc.pl.violin(filtered_not_targeted, keys=marker_gene, groupby="group_cell_type", rotation=90, ax=axs[0], show=False, use_raw=False)
            axs[0].set_title(f"{sample} not_targeted, score={str(round(score,2))}")
            
            score = signal_to_noise_ratio(data_samples, marker_gene, dataset=f"{sample}_targeted", median=median, sparse=sparse)
            filtered_targeted = data_samples[f"{sample}_targeted"][data_samples[f"{sample}_targeted"].obs["group_cell_type"] != "Excluded cells"].copy()
            sc.pl.violin(filtered_targeted, keys=marker_gene, groupby="group_cell_type", rotation=90, ax=axs[1], show=False, title=f"targeted", use_raw=False)
            axs[1].set_title(f"{sample} targeted, score={str(round(score,2))}")
            
            score = signal_to_noise_ratio(data_samples, marker_gene, dataset=f"{sample}_combined", median=median, sparse=sparse)
            filtered_combined = data_samples[f"{sample}_combined"][data_samples[f"{sample}_combined"].obs["group_cell_type"] != "Excluded cells"].copy()
            sc.pl.violin(filtered_combined, keys=marker_gene, groupby="group_cell_type", rotation=90, ax=axs[2], show=False, title=f"combined", use_raw=False)
            axs[2].set_title(f"{sample} combined, score={str(round(score,2))}")
            
            savefilename = f"{save_directory}/violin_plot_SNR_{sample}_{marker_gene}.pdf"
            fig.tight_layout()
            fig.savefig(savefilename)
            plt.close("all")
            
def calculate_snr_scores(data_samples, marker_genes, target_cell_types_list, excluded_cell_types_list, samples, save_directory, cell_type_annotation="Smooth_cell_type", sample_word_number=1, median=True, sparse=False, samples_to_exclude=None):
    """
    Function to calculate a DataFrame of SNR (Signal-to-Noise Ratio) scores of marker genes in target cell types.
    
    excluded_cell_types_list: cell types that are not target cell types, but also not fully negative cell types. They might have a subpopulation that expressed the marker.
    We exclude them from the signal-to-noise ratio calculation.
    """
    marker_dict = {}
    for i in range(len(marker_genes)):
        marker_gene = marker_genes[i]
        target_cell_types = target_cell_types_list[i]
        excluded_cell_types = excluded_cell_types_list[i]
        dataset_dict = {}
        for dataset in data_samples.keys():
            sample = "_".join(dataset.split("_")[:sample_word_number])
            if samples_to_exclude:
                if sample in samples_to_exclude[i]:
                    continue
            if (sample in samples) and (("RNA" not in dataset) or ("KNN" in dataset)):
                data_samples[dataset].obs["group_cell_type"] = data_samples[dataset].obs[cell_type_annotation].apply(lambda x: "Target cells" if x in target_cell_types else ("Excluded cells" if x in excluded_cell_types else "Other cells"))
                score = signal_to_noise_ratio(data_samples, marker_gene, dataset, median=median, sparse=sparse)
                dataset_dict[dataset] = score
                
        marker_dict[marker_gene] = dataset_dict
    
    marker_df = pd.DataFrame(marker_dict)
    marker_df_simplified = average_across_samples(marker_df, savefilename=f"{save_directory}/snr_scores_simplified.csv", sample_word_number=sample_word_number)
    marker_df["average"] = marker_df.mean(axis=1).round(3)
    marker_df = marker_df[['average'] + [col for col in marker_df.columns if col != 'average']] # Re-order to put average first.
    marker_df.round(3).to_csv(f"{save_directory}/snr_scores.csv")

    return marker_df, marker_df_simplified

def calculate_and_plot_marker_snr(data_samples, cell_type_annotation, marker_genes, target_cell_types_list, excluded_cell_types_list, 
                                  samples_to_average, samples_to_exclude, working_directory, sample_word_number, 
                                  selected_dataset_and_methods, methods_palette, profi, orientation="vertical", xticks=None):

    snr_scores, snr_scores_simplified = calculate_snr_scores(data_samples, marker_genes = marker_genes, 
                   target_cell_types_list=target_cell_types_list, excluded_cell_types_list=excluded_cell_types_list, 
                   samples = samples_to_average, save_directory = f"{working_directory}/figures/marker_gene_scores",
                  cell_type_annotation=cell_type_annotation, sample_word_number = sample_word_number, samples_to_exclude=samples_to_exclude)

    # Simplified barplot to compare the SNR by normalization method.
    snr_scores_super_simplified = snr_scores_simplified.mean(axis=1)
    snr_scores_super_simplified.name = "marker_signal"
    snr_scores_super_simplified.round(3).to_csv(f"{working_directory}/figures/marker_gene_scores/snr_scores_super_simplified_{cell_type_annotation}.csv")
    snr_scores_super_simplified = snr_scores_super_simplified.reset_index().rename(columns={"index": "assay"})
    plot_simplified_table(snr_scores_super_simplified, title=f"Simplified marker signal-to-noise ratio",
                     savefilename=f"{working_directory}/figures/marker_gene_scores/barplot_marker_gene_scores_super_simplified_{cell_type_annotation}.pdf")

    boxplot_comparing_normalizations(snr_scores, savefilename=f"{working_directory}/figures/marker_gene_scores/SNR_{cell_type_annotation}_by_method.pdf", 
                                     profi=profi, metric="SNR", selected_dataset_and_methods=selected_dataset_and_methods, 
                                     sample_word_number=sample_word_number, methods_palette=methods_palette, orientation=orientation, xticks=xticks)


def average_across_samples(df, savefilename, sample_word_number=1):
    """
    Function to average a dataframe across samples. I assume that the assay name is the index of the dataframe, and the first words
    of the assay name are the sample.
    """
    
    # Across what do we want to calculate the mean? Across samples, that is, take everything except the first word (the samples).
    levels = df.index.str.split('_').str[sample_word_number:].str.join('_')
    simplified_df = df.groupby(levels, sort=False).mean() # Take the mean across the samples.
    simplified_df.round(3).to_csv(savefilename)
    
    return simplified_df


def plot_simplified_table(df, title, savefilename):
    """
    Function to plot a simplified table as a barplot.
    """
    simp_barplot = sns.barplot(data=df, x=df.columns[-1], y="assay", hue="assay", legend=False, orient="y")
    simp_barplot.tick_params(left=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savefilename)
    plt.close("all")


def library_normalize_total(anndata, target_sum=None):
    """
    Function to perform library normalization. Based on sc.pp.normalize_total().
    """
    X = anndata.X
    if issubclass(X.dtype.type, (int, np.integer)):
        X = X.astype(np.float32)  # TODO: Check if float64 should be used
    
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


def obs_names_make_unique_by_sample(anndata, sample_column_name="sample", remove_dash_suffix=True):
    """
    Function to change obs names in a way that the final name is cellbarcode_sample (or cellbarde_libraryname).
    
    remove_dash_suffix: whether to remove the suffix from the cell barcode. Often obs_names are called something like
    "TTTGTTGGTTGTACGT-1". This argument will remove the "-1"-
    """
    if remove_dash_suffix:
        anndata.obs["cell_barcode"] = [name.split("-")[0] for name in anndata.obs_names]
    else:
        anndata.obs["cell_barcode"] = anndata.obs_names
    anndata.obs["cell_barcode_sample"] = anndata.obs.apply(lambda row: row["cell_barcode"] + "_" + row[sample_column_name], axis=1)
    anndata.obs_names = anndata.obs["cell_barcode_sample"]


def model_function(mean, alpha):
    return mean + alpha * mean**2


def fit_alpha(means, variances):
    popt, pcov = curve_fit(model_function, means, variances)
    alpha = popt[0]
    return alpha


def calculate_heteroskedasticity(data, sparse=False):
    
    if sparse:
        means, variances = sc.pp._utils._get_mean_var(data.X)
    else:
        means = data.X.mean(axis=0)
        variances = data.X.var(axis=0)
        
    alpha = fit_alpha(means, variances)
    
    return alpha, means, variances


def plot_mean_variance_with_fit(data, savefilename, title, sparse=False):
    
    alpha, means, variances = calculate_heteroskedasticity(data, sparse=sparse)
    
    fig, ax = plt.subplots()
    ax.scatter(means, variances, label='Data')
    
    # Fit a regression line
    # m, b = np.polyfit(np.log(variances),np.log(means), 1)
    # x_fit = np.linspace(min(variances), max(variances), 100)
    # y_fit = np.exp(m * np.log(x_fit) + b)
    # ax.plot(x_fit, y_fit, color='red')
    
    # Plotting the model curve
    model_means = np.linspace(min(means), max(means), 500)
    model_variances = model_function(model_means, alpha)
    ax.plot(model_means, model_variances, color='red', label=f'Model: Var[Y] = µ + {alpha:.2f}µ^2')
    
    plt.xlabel("mean")
    plt.ylabel("variance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.title(title)
    plt.legend()
    plt.savefig(savefilename)
    plt.show()
    plt.close("all")


def boxplot_comparing_normalizations(df, savefilename, methods_palette, sample_word_number, selected_dataset_and_methods, 
                                     profi=True, metric="SNR", figsize=(12,6), 
                                     rcparams = {'font.size': 19, 'axes.titlesize': 22, 'axes.labelsize': 16, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14, 'figure.titlesize': 22}, 
                                     orientation="vertical", xticks=None):

    """Boxplot to compare the ARI, silhouette, SNR, etc. by normalization method."""
    
    df["sample"] = df.index.str.split("_").str[:sample_word_number].str.join("_")
    df["normalization"] = df.index.str.split("_").str[sample_word_number:].str.join("_")
    if profi:
        df = df[df["normalization"].isin(selected_dataset_and_methods.keys())].copy()
    df["normalization"] = df["normalization"].replace(selected_dataset_and_methods)
    
    if metric=="SNR":
        df = df.melt(id_vars=["normalization", "sample"], value_vars=df.columns[1:-2], var_name="Marker", value_name="SNR") # Melt for boxplot (long format)

    plt.rcParams.update(rcparams)
    plt.figure(figsize=figsize)
    
    if profi:
        # Define the order of categories for consistent plotting
        method_order = list(selected_dataset_and_methods.values())
        
        # Assign colors based on the mapping
        palette = {method: methods_palette[method] for method in method_order if method in methods_palette}
        if orientation=="vertical":
            sns.boxplot(x="normalization", y=metric, data=df, order=method_order, palette=palette, hue="normalization")
        else:
            sns.boxplot(x=metric, y="normalization", data=df, order=method_order, palette=palette, hue="normalization", orientation="horizontal")
    else:
        if orientation=="vertical":
            sns.boxplot(x="normalization", y=metric, data=df, hue="normalization")
        else:
            sns.boxplot(x=metric, y="normalization", data=df, hue="normalization", orientation="horizontal")
    plt.xticks(rotation=90)
    dict_metric_name = {"SNR": "Signal to Noise Ratio", "ari": "Adjusted Rand Index", "ARI": "Adjusted Rand Index", "nmi": "Normalized Mutual Information", "silhouette": "Silhouette score", "knn_classifier_accuracy": "KNN Classifier Accuracy"}
    if orientation=="vertical":
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


def neighborhood_analysis(normalized_data, palette, cell_type_annotation = "Cell type", save_directory = "figures"):

    for dataset in normalized_data:
        if "KNN" in dataset:
            protein_knn_normalized = normalized_data[dataset]
    
            # KNN graph used in the normalization: row i holds the neighbors of cell i (explicit zeros are removed).
            connectivities = protein_knn_normalized.obsp["connectivities_KNN_normalization"].tocsr(copy=True)
            connectivities.eliminate_zeros()
            total_counts_neighbors = {}

            for cell_type in protein_knn_normalized.obs[cell_type_annotation].cat.categories:
                celltype_index = (protein_knn_normalized.obs[cell_type_annotation] == cell_type).to_numpy().nonzero()[0] # Indices of cell from this cell type
                flattened_celltype_neighbors = connectivities[celltype_index].indices # Neighbors of the cells of this cell type.
                celltype_neighbors_counts = protein_knn_normalized[flattened_celltype_neighbors].obs[cell_type_annotation].value_counts() # Cell types of neighbors.
                
                sns.barplot(x=celltype_neighbors_counts.index, y=celltype_neighbors_counts.values, hue = celltype_neighbors_counts.index, palette = palette)
                plt.title(f"Neighbors by cell type for {cell_type} cells in {dataset}")
                plt.ylabel("Number of Neighbors")
                plt.xlabel("Cell Type")
                plt.savefig(f"{save_directory}/barplot_{dataset}_{cell_type}_neighbors_celltype.pdf")
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


def calculate_silhouette_score(data_samples, cell_annotation_levels, samples_to_average, sample_word_number, working_directory,
                               profi, selected_dataset_and_methods, methods_palette, use_pca_silhouette, subsample_silhouette, 
                               rcparams, figsize=(2.12, 1.60), orientation="vertical", xticks=None):
    for cell_type_annotation in cell_annotation_levels:
        silhouette_dict = {}
        for dataset in data_samples.keys():
            if "_".join(dataset.split("_")[:sample_word_number]) in samples_to_average: 
                if (use_pca_silhouette or (("RNA" in dataset) and ("KNN" not in dataset))):
                    if "X_pca" not in data_samples[dataset].obsm:
                        data_silhouette = PCA(n_components=50).fit_transform(data_samples[dataset].X)
                    else:
                        data_silhouette = data_samples[dataset].obsm["X_pca"]   
                else:
                    data_silhouette = data_samples[dataset].X
                    
                if subsample_silhouette: # Subsample cells to reduce compute time.
                    n_cells = data_samples[dataset].shape[0]
                    subset_indices = np.random.choice(n_cells, size=min(1000, n_cells), replace=False)
                    data_silhouette = data_silhouette[subset_indices]
                    cell_type_annotations = data_samples[dataset].obs[cell_type_annotation].iloc[subset_indices]
                else:
                    cell_type_annotations = data_samples[dataset].obs[cell_type_annotation]
                
                silhouette = silhouette_score(data_silhouette, cell_type_annotations)
                silhouette_dict[dataset] = silhouette
        silhouette_df = pd.DataFrame(silhouette_dict, index=["silhouette"]).T
        silhouette_df.round(3).to_csv(f"{working_directory}/figures/silhouette_scores_{cell_type_annotation}.csv", index=True)
        silhouette_df_simplified = average_across_samples(silhouette_df,
                                                                    f"{working_directory}/figures/silhouette_scores_simplified_{cell_type_annotation}.csv",
                                                                    sample_word_number = sample_word_number)
        silhouette_df_simplified = silhouette_df_simplified.reset_index().rename(columns={"index": "assay"})
        boxplot_comparing_normalizations(silhouette_df, savefilename=f"{working_directory}/figures/silhouette_{cell_type_annotation}_by_method", 
                                         profi=profi, metric="silhouette", selected_dataset_and_methods=selected_dataset_and_methods, 
                                         sample_word_number=sample_word_number, methods_palette=methods_palette, figsize=figsize, 
                                         rcparams=rcparams, orientation=orientation, xticks=xticks)

def remove_very_high_total_counts_by_sample(protein_data, rna_data, sample_column_name, percentile):
    """
    Function to remove a certain pertencile of cells with highest total protein counts. Removes outliers with very high total counts.
    """
    for sample in protein_data.obs[sample_column_name].unique():
        counts_percentile = np.percentile(protein_data[protein_data.obs[sample_column_name] == sample].obs["total_counts"], percentile)
        cells_to_remove = (protein_data.obs[sample_column_name] == sample) & (protein_data.obs["total_counts"] >= counts_percentile)
        protein_data = protein_data[~cells_to_remove].copy()
        rna_data = rna_data[~cells_to_remove].copy()
    return protein_data, rna_data