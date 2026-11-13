library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(scMultiSim)
library(ape)

create_seurat_from_scmultisim = function(scmultisim_data, use_obs = FALSE, protein=TRUE){
    
    counts_matrix = if (use_obs) scmultisim_data$counts_obs else scmultisim_data$counts

    colnames(counts_matrix) = paste0("cell_", 1:ncol(counts_matrix))
    rownames(counts_matrix) = if (protein) paste0("protein-", 1:nrow(counts_matrix))  else paste0("gene-", 1:nrow(counts_matrix))
    rownames(scmultisim_data$cell_meta) = scmultisim_data$cell_meta$cellid
    seurat_object = CreateSeuratObject(counts = counts_matrix, meta.data = scmultisim_data$cell_meta)
    return(seurat_object)
}
# This function converts only the counts, which is all we need anyway.
convert_seurat_v5_to_v3 = function(seurat_v5_object){
    rna3_assay <- CreateAssayObject(counts = LayerData(seurat_v5_object, layer = "counts"))
    # rna3_assay@data <- LayerData(seurat_v5_object, layer = "data"
    # rna3_assay@scale.data <- LayerData(seurat_v5_object, layer = "scale.data"
    srat <- CreateSeuratObject(counts = rna3_assay@counts)
    srat@meta.data <- seurat_v5_object@meta.data
    srat[["RNA3"]] <- rna3_assay
    DefaultAssay(srat) <- "RNA3"
    srat[["RNA"]] <- NULL
    srat[["RNA"]] <- srat[["RNA3"]]
    DefaultAssay(srat) <- "RNA"
    srat[["RNA3"]] <- NULL
    return (srat)
}
export_seuratv5_to_scanpy = function(seuratv5_object, filename){
    seurat_v3 = convert_seurat_v5_to_v3(seuratv5_object)
    SaveH5Seurat(seurat_v3, filename = paste0(filename, ".h5Seurat"))
    Convert(paste0(filename, ".h5Seurat"), dest = paste0(filename, ".h5ad"))
    }
ngenes = 5000
nproteins = 50
ncells = 1000
ncelltypes = 3
true_counts_rna = sim_true_counts(list(GRN = NA, 
                                       rand.seed = 0, 
                                       num.cells = ncells, 
                                       num.genes = ngenes, 
                                       tree=ape::read.tree(text = "(A:1,B:1,C:1);"), 
                                       num.cifs = 150, 
                                       discrete.cif = TRUE, 
                                       cif.sigma = 0.8, 
                                       scale.s=c(1, 1, 1), 
                                       do.velocity=F,
                                       discrete.pop.size=as.integer(c(333, 333, 334))
                                      ))
true_counts_rna_seurat = create_seurat_from_scmultisim(true_counts_rna, protein = FALSE)
true_counts_protein = sim_true_counts(list(GRN = NA, 
                                       rand.seed = 0, 
                                       num.cells = ncells, 
                                       num.genes = nproteins, 
                                       tree=ape::read.tree(text = "(A:1,B:1,C:1);"), 
                                       num.cifs = 150, 
                                       discrete.cif = TRUE, 
                                       scale.s=c(2, 5, 12), 
                                       do.velocity=FALSE,
                                       discrete.pop.size=as.integer(c(333, 333, 334)),
                                       cif.sigma = 0.3,
                                       intrinsic.noise = 0.5
                                      ))
true_counts_protein_seurat = create_seurat_from_scmultisim(true_counts_protein)
export_seuratv5_to_scanpy(true_counts_protein_seurat, "scmultisim_protein_data")
export_seuratv5_to_scanpy(true_counts_rna_seurat, "scmultisim_rna_data")