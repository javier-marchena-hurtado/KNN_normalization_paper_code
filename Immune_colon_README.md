# Immune colon CITE-seq dataset (protein and RNA)

Paired protein (ADT) and RNA counts of 4,075 immune cells from colorectal cancer samples, stored as AnnData (`.h5ad`) files. They were used in the paper **[paper title]** ([authors], [year], [DOI]) to benchmark KNN normalization of protein data. Analysis code: [GitHub URL] and [Zenodo DOI of the code record].

Original data source: [citation of the original study and accession, if applicable].

## Files

| File | Shape (cells x features) | Description |
|---|---|---|
| `Immune_colon_protein_data.h5ad` | 4,075 x 61 | Protein (antibody-derived tag) data. |
| `Immune_colon_rna_data.h5ad` | 4,075 x 25,154 | RNA data. |

Both files contain the same cells in the same order (`obs_names` are identical).

## Contents of each file

Cell metadata (`.obs`, identical in both files):

| Column | Description |
|---|---|
| `sample` | Sample identifier (7 samples, e.g. `p026t`). |
| `patient` | Patient identifier (4 patients). |
| `origin` | Tissue of origin of the sample (`t` = tumor, `n` = normal in the sample names). |
| `RNA-based_cell_type_coarse`, `RNA-based_cell_type_detailed` | Cell type annotations based on RNA (8 coarse types: CD4 T, CD8 T, Treg, NK, B, Plasma, Myeloid, pDC). |
| `CITE-seq-based_cell_type_coarse`, `CITE-seq-based_cell_type_detailed` | Cell type annotations based on protein data. |
| `TCR`, `clonal_expansion`, `clone_size`, `clone_id` | T-cell receptor clonotype information. |
| `cytotoxicity_signature`, `exhaustion_signature` | T-cell signature scores. |

Protein file:
- `X`: protein counts.
- `layers["KNN_normalized_w_RNA_neighbors"]`: protein data after KNN normalization with neighbors defined from RNA.
- `layers["KNN_normalized_w_RNA_neighbors_and_scaled_per_sample"]`: the same, scaled per sample.
- `obsm["KNN_scaled_CytoVI_embedding"]`: CytoVI latent embedding of the KNN-normalized, scaled data; `obsm["UMAP_KNN_scaled_CytoVI_embedding"]` is its UMAP.
- `obsm["X_umap"]`: UMAP.

RNA file:
- `X`: RNA counts.
- `layers["cellranger_counts"]`: counts as output by Cell Ranger.
- `layers["normalized_and_log"]`: library-size normalized, log-transformed.
- `layers["normalized_log_and_scaled_per_sample"]`: the same, scaled per sample.
- `var`: gene annotation (`gene_id`, `source`, `chromosome`, `start`, `end`, `strand`).
- `obsm["X_umap"]`: UMAP.

## Usage

```python
import scanpy as sc

protein = sc.read_h5ad("Immune_colon_protein_data.h5ad")
rna = sc.read_h5ad("Immune_colon_rna_data.h5ad")
```

KNN normalization is implemented in the [knn_normalization]([package URL]) package.

## License and citation

Released under [license]. If you use these data, please cite [the paper] and [the original study].
