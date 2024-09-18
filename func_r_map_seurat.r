#!/usr/bin/env python
# coding: utf-8

# # func_r_map_Seurat
# 
# 最后更新时间2024年4月11日
# 
# Seurat 5.0.1的标签转移和UMAP坐标映射
# 
# 其中`Map_Seurat_normalize`和`Map_Seurat_cluster`已经实现了单细胞处理的标准流程
# 
# 
# + Seurat_to_mtx
# + Map_Seurat_normalize
# + Map_Seurat_cluster
# + Map_Seurat_cluster_run_harmony
# + Map_Seurat_mapquery
# + Map_Seurat_example
# + precess_after_Seurat
# + run_Seurat
# 
# ```bash
# conda activate
# {
# cd ~/link/res_publish
# jupyter nbconvert func_r_map_Seurat.ipynb --to python
# mv func_r_map_Seurat.py func_r_map_Seurat.r
# echo "finish"
# }
# ```
# 

# In[1]:


library(tidyverse)
library(Seurat)
library(Matrix)
library(harmony)
library(IRdisplay)


p_root = file.path('~/link/res_publish')
p_run = file.path(p_root,'run')
p_res = file.path(p_root,'res')
p_cache = file.path(p_run,'cache')
p_df_varmap = file.path(p_root,"homo/df_varmap.csv")
if(! file.exists(p_df_varmap)){
    stop(sprintf('[not exists] %s\n',p_df_varmap))
}


# # other

# In[2]:


get_path_varmap <- function(q_sp_ref, q_sp_que, df_varmap_path = p_df_varmap ) {
    df_varmap <- read.csv(df_varmap_path)
    filtered_rows <- df_varmap %>%
        filter(sp_ref == q_sp_ref & sp_que == q_sp_que)
    
    # 唯一匹配项
    if (! nrow(filtered_rows) == 1) {
        stop("[get path] Cannot get specified and unique path\nq_sp_ref\tq_sp_que\n", q_sp_ref, "\t", q_sp_que)
    }
    path_varmap <- file.path(filtered_rows$path[1])
    # isAbsolut
    if(! str_detect('^/',path_varmap)){
        path_varmap <- file.path(dirname(p_df_varmap),path_varmap)
    }
    # 存在
    if (!file.exists(path_varmap)) {
        stop("[not exists] ", as.character(path_varmap))
    }

    return(path_varmap)

}


show_obj = function(obj,tag = ''){
    cat(sprintf('%s--------------------\n',tag))
    cat(sprintf("\t%s %s\n",typeof(obj),class(obj)))
    if(typeof(obj) == 'S4'){cat('\t@',slotNames(obj),'\n')}
    if(obj %>% names %>% length > 5){
        cat('\t$',names((obj)) %>% head(5) ,
        sprintf("[names lenght] %d",
        obj %>% names %>% length) ,'\n')    
    }else{
        cat('\t$',names((obj)),'\n')    
    }
    
}

savefig  <- function(fig, fig_name, p_plot = p_plot,
                     height_ratio = 4, width_ratio = 4) {
  if (!dir.exists(p_plot)) {
    stop(sprintf("[Error][not exists] %s", p_plot))
  }
  file.path(p_plot, fig_name) %>%
    ggsave(fig, dpi = 200, bg = "transparent",
      width = 200 * width_ratio, height = 200 * height_ratio,
      units = "px")
  cat(sprintf("[out][plot] %s
\tin %s
", fig_name, p_plot))
}

df.iterdir = function(p,path_match=NULL,path_match_filter=c(),select_dir=FALSE){
    df = tibble(file_name = list.files(p),
            file_path = file.path(p,file_name)) %>% select(file_path,file_name)
    if(select_dir){
        df  = df %>% filter(dir.exists(file_path))
    }else{
        df  = df %>% filter(!dir.exists(file_path))
    }
    
    
    for(pattern in path_match_filter){
        df = df %>% filter(!str_detect(file_path,pattern))
    }
    if(!is.null(path_match)){
        df = df %>% filter(
            str_detect(file_path,path_match)
        )    
    }
    return(df)
}

df.show  = function(df,nrows = 2){
    display(df %>% head(nrows))
    display(df %>% dim)
}


# In[3]:


Seurat_metadata_leftjoin = function(metadata,join_data,by,key_cell_name='cell_name'){
    if(! key_cell_name %in% colnames(metadata)){
        stop(sprintf("[Error] '%s' not in metadata\n",key_cell_name))
    }
    metadata = metadata %>% left_join(join_data,by = by)
    rownames(metadata) = metadata[[key_cell_name]]
    return(metadata)
}

Seurat_gene_detect = function(adata,detect_regex){
    temp = adata@assays$RNA@features %>% as.data.frame
    temp = temp %>% mutate( gene = rownames(temp))
    temp = temp %>% filter(str_detect(gene,detect_regex))
    return(temp)
}

Seurat_load_reductions_from_metadata  = function(metadata,
                                                 key_metadata=c('UMAP1','UMAP2'),
                                                 key_reductions='umap_'){
    # UMAP 的key 为 umap_
    # tSN 的key 为 tSNE_
    # ------------------------------
    # 调用
    # ------------------------------
    # adata@reductions$umap = Seurat_load_reductions_from_metadata(adata@meta.data,
    #     key_metadata=c('UMAP1','UMAP2'),key_reductions='umap_')
    
    # adata@reductions$tsne = Seurat_load_reductions_from_metadata(adata@meta.data,
    #     key_metadata=c('tSNE_1','tSNE_2'),key_reductions='tSNE_')
    
    return(CreateDimReducObject(
        embeddings=metadata %>% select(any_of(key_metadata))%>% as.matrix,
        key=key_reductions))

}


# # Seurat IO

# In[ ]:


load_Seuratobj_add_obs <- function(p, adata, key_cell_name = "cell_name") {
  if (file.exists(file.path(p, "obs.csv"))) {
    df_obs <- read.csv(file.path(p, "obs.csv"), row.names = 1)
    df_obs <- df_obs %>%
      mutate(cell_name = rownames(df_obs))
    adata@meta.data <- adata@meta.data %>%
      mutate(cell_name = rownames(adata@meta.data), .before = 1) %>%
      left_join(df_obs, by = c(cell_name = key_cell_name)) %>%
      as.data.frame()
    # left_join 返回tibble , index丢失了，加回来
    rownames(adata@meta.data) <- adata@meta.data$cell_name

  } else {
    display_text(sprintf("[not obs.csv] %s\n", p))
  }
  return(adata)
}

load_Seuratobj <- function(p, return_matrix = FALSE, add_obs = TRUE, key_cell_name = "cell_name") {
  adata = NULL
  if( file.exists( file.path(p,'matrix.mtx.gz'))){
      system(sprintf("gzip -dc %s > %s ",file.path(p,'matrix.mtx.gz'),file.path(p,'matrix.mtx')))
      adata <- Read10X(p)
      system(sprintf('gzip %s ',file.path(p,'matrix.mtx')))
  }else{
      adata <- Read10X(p)
  }
    
  if (adata %>% colnames() %>% str_detect('"([^"]+)"') %>% all()) {
    colnames(adata) <- str_extract(colnames(adata), '"([^"]+)"', group = 1)
  }
  if (return_matrix) return(adata)
  adata <- CreateSeuratObject(adata)
  if (add_obs) {
    adata <- load_Seuratobj_add_obs(p, adata, key_cell_name = key_cell_name)
  }
  return(adata)
}


# In[4]:


Seurat_to_mtx <- function(adata, p_dir, prefixes = "") {
  if (!dir.exists(p_dir)) {
    dir.create(p_dir, recursive = T)
  }

  sparse <- Matrix(adata@assays$RNA@layers$counts, sparse = T)

  # [out] genes.tsv
  df_genes <- adata@assays$RNA@features %>% as.data.frame()
  df_genes <- df_genes %>%
    transmute(
      gene_names = rownames(df_genes),
      gene_ids = ifelse("gene_ids" %in% colnames(df_genes), gene_ids, gene_names)
    ) %>%
    select(gene_ids, gene_names)
  df_genes %>% write.table(file.path(p_dir, sprintf("%sgenes.tsv", prefixes)),
    row.names = F, col.names = F, sep = "\t", fileEncoding = "utf-8"
  )

  # [out] barcodes.tsv obs.csv
  tibble(barcodes = rownames(adata@assays$RNA@cells)) %>% write.table(file.path(p_dir, sprintf("%sbarcodes.tsv", prefixes)),
    row.names = F, col.names = F, sep = "\t", fileEncoding = "utf-8"
  )
  adata@meta.data = adata@meta.data %>% select(-matches("(orig.ident)|(nCount_RNA)|(nFeature_RNA)"))
  if (adata@meta.data %>% colnames() %>% length() > 0) {
    adata@meta.data %>% write.csv(file.path(p_dir, sprintf("%sobs.csv", prefixes)),
      row.names = T,
      # col.names = T,
      fileEncoding = "utf-8"
    )
  }

  # [out] matrix.mx
  writeMM(sparse, file = file.path(p_dir, sprintf("%smatrix.mtx", prefixes)))
  cat(sprintf("[out] %s\n", p_dir))
}
seurat_to_mtx = Seurat_to_mtx


# # Map Seurat corss speciese
# 
# ## Method of came's article
# 
# For Seurat V3, we
# 
# + input the raw data;
# + used the default normalize process by NormalizeData() function;
# + extracted the **top 2000 HVGs** by its FindVariableFeatures() function for reference and query, respectively;
# + 【？】and performed further annotation process as described in its documentation 

# # Seurat flow
# 
# ## Map_Seurat_normalize

# In[5]:


Map_Seurat_normalize <- function(adata, hvg_nfeatures = 2000, run_scale = TRUE, run_pca = TRUE, verbose = FALSE) {
  # R 的形参不是引用，而是完全复制了一份
  adata <- NormalizeData(adata, verbose = verbose)
  adata <- FindVariableFeatures(adata, verbose = verbose, nfeatures = hvg_nfeatures)
  if (run_scale) {
    adata <- ScaleData(adata, verbose = verbose)
  }
  if (run_scale & run_pca) {
    adata <- RunPCA(adata, verbose = verbose)
    print(ElbowPlot(adata, ndims = 50))
  }
  return(adata)

}


# ## Map_Seurat_cluster

# In[6]:


Map_Seurat_cluster =  function(adata,dims,resolution,key_celltype=NULL,verbose=FALSE){
    adata <- FindNeighbors(adata,
      dims = dims, verbose = verbose
    )
    adata <- FindClusters(adata, resolution = resolution, verbose = verbose)
    # 返回umap model，后续的Running UMAP projection 需要umap model
    adata <- RunUMAP(adata, dims = dims,verbose = verbose,return.model = TRUE)
    if(!is_null(key_celltype)){
        print(DimPlot(adata, group.by = c(key_celltype), reduction = "umap"))    
    }
    print(DimPlot(adata, group.by = c("seurat_clusters"), reduction = "umap"))
    
    df_umap = adata@reductions$umap@cell.embeddings %>% as.data.frame %>%
        rownames_to_column('cell_name')
    colnames(df_umap) = str_split('cell_name,UMAP1,UMAP2',',')[[1]]
    adata@meta.data = adata@meta.data %>% select(-matches('^UMAP')) %>%
        Seurat_metadata_leftjoin(df_umap,by='cell_name')
    return(adata)
}


# ## Map_Seurat_cluster_run_harmony

# In[7]:


them_legend <- theme(
  legend.position = "inside",
  legend.justification = c(0, 0),
    rect = element_rect(fill = "transparent")
)
Map_Seurat_cluster_run_harmony =  function(
    adata,dims,
    resolution,key_batch,
    key_celltype=NULL,verbose=FALSE
){
    if(! key_batch %in% colnames(adata@meta.data)){
        stop(sprintf("[Error] key_batch = %s is not in adata@meta.data",key_batch))
    }

    adata@meta.data %>% count(.data[[key_batch]])
    adata <- adata %>% RunHarmony(key_batch, plot_convergence = TRUE, 
                                  return.model = TRUE, verbose = verbose)
    print(DimPlot(object = adata, reduction = "pca", group.by = key_batch,
                  pt.size =2e4/nrow(adata@meta.data),) + them_legend)
    print(DimPlot(object = adata, reduction = "harmony", group.by = key_batch,
                  pt.size =2e4/nrow(adata@meta.data)) + them_legend)
    
    adata <- adata %>%
      RunUMAP(reduction = "harmony", dims = dims, verbose = verbose) %>%
      FindNeighbors(reduction = "harmony", dims = dims, verbose = verbose) %>%
      FindClusters(resolution = resolution, verbose = verbose)
    
    df_umap = adata@reductions$umap@cell.embeddings %>% as.data.frame %>%
        rownames_to_column('cell_name')
    colnames(df_umap) = str_split('cell_name,UMAP1,UMAP2',',')[[1]]
    adata@meta.data = adata@meta.data %>% select(-matches('^UMAP')) %>%
        Seurat_metadata_leftjoin(df_umap,by='cell_name')
    return(adata)
}


# # Map_Seurat_mapquery

# In[8]:


# refdata =list(predicted.id = "CL_cell_subtype1")
Map_Seurat_mapquery <-  function(adata_ref, adata_que, dims, refdata, reference.reduction = "pca",
    verbose = FALSE

    ) {
  adata.anchors <- FindTransferAnchors(
    reference = adata_ref, query = adata_que, dims = dims,
    reference.reduction = reference.reduction, verbose = verbose
  )
  adata_que <- MapQuery(anchorset = adata.anchors, reference = adata_ref, query = adata_que,
    refdata = refdata, reference.reduction = reference.reduction, reduction.model = "umap", verbose = verbose)
  return(adata_que)
}


# ### F1-score
# 
# ```r
# # 使用
# cm = calculate_more_with_confusion_matrix(
#     calculate_confusion_matrix(actual,predicted)
# )
# cm
# calculate_accuracy_with_confusion_matrix(cm)
# calculate_F1Score_with_confusion_matrix(cm)
# ```

# In[9]:


# 计算混淆矩阵的函数  
calculate_confusion_matrix <- function(y_true, y_pred) {  
  classes <- sort(unique(c(y_true, y_pred)))  
  matrix <- matrix(0, nrow = length(classes), ncol = length(classes), dimnames = list(classes, classes))  
    
  for (i in seq_along(y_true)) {  
    matrix[y_true[i], y_pred[i]] <- matrix[y_true[i], y_pred[i]] + 1  
  }  
    
  return(as.data.frame(matrix))  
}  
  
# 计算精确度、精确度和召回率等指标的函数
calculate_more_with_confusion_matrix <- function(data) {  
    # data 为 calculate_confusion_matrix的运行结果
  TP <- diag(as.matrix(data))  
  FP <- colSums(data) - TP  
  FN <- rowSums(data) - TP  
  Precision <- TP / (TP + FP)  
  Recall <- TP / (TP + FN)  
  F1_Score <- 2 * (Precision * Recall) / (Precision + Recall)  
    
  df <- data.frame(TP, FP, FN,  
                    Precision = ifelse(is.nan(Precision), 0, Precision),  
                    Recall = ifelse(is.nan(Recall), 0, Recall),  
                    F1_Score = ifelse(is.nan(F1_Score), 0, F1_Score))  
    
  return(df)  
}  

calculate_accuracy_with_confusion_matrix <- function(data) {  
    # data 为 calculate_more_with_confusion_matrix的运行结果
  accuracy <- sum(diag(as.matrix(data))) / sum(data)  
  return(accuracy)  
}

calculate_F1Score_with_confusion_matrix <- function(data, average = "weighted") {  
    # data 为 calculate_more_with_confusion_matrix的运行结果
    
  # 确保data是一个数据框，并且包含'TP', 'FP', 'FN'和'F1 Score'列  
  if (!is.data.frame(data) || !all(c("TP", "FP", "FN", "F1_Score") %in% names(data))) {  
    stop("The data should be a data frame with columns 'TP', 'FP', 'FN', and 'F1_Score'.")  
  }  
    
  # 根据average参数计算F1分数  
  if (average == "macro") {  
    res <- mean(data$F1_Score)  
  } else if (average == "weighted") {  
    weights <- data$TP + data$FN  
    res <- sum(data$F1_Score * weights) / sum(weights)  
  } else if (average == "micro") {  
    res <- 2 * sum(data$TP) / (2 * sum(data$TP) + sum(data$FP) + sum(data$FN))  
  } else {  
    stop(paste("[Error] Invalid average parameter:", average))  
  }  
    
  return(res)  
} 


# ### precess_after_Seurat

# In[10]:


precess_after_Seurat <- function(resdir, adata_ref, adt, adata_que, key_celltype, tissue_name, sp1, sp2) {
  # adt@meta.data %>% head(2)
  df_obs <-  bind_rows(

    # ref
    tibble(
      cell_name = rownames(adata_ref@meta.data),
      dataset = paste(tissue_name, sp1, sep = "_"),
      cell_type = adata_ref@meta.data[[key_celltype]],
      true_label = adata_ref@meta.data[[key_celltype]],
      pre_label = rep(NA, adata_ref %>% Cells() %>% length()),
      max_prob = rep(NA, adata_ref %>% Cells() %>% length()),
      is_right = rep(NA, adata_ref %>% Cells() %>% length())
    ),
    # que
    tibble(
      cell_name = rownames(adt@meta.data),
      dataset = paste(tissue_name, sp2, sep = "_"),
      cell_type = adt@meta.data[[key_celltype]],
      true_label = adt@meta.data[[key_celltype]],
      pre_label = adt@meta.data[["predicted.."]],
      max_prob = adt@meta.data[["predicted...score"]],
      is_right = (true_label == pre_label)
    )
  )

    df_umap <- bind_rows(
      adata_ref@reductions$uma@cell.embeddings %>% as.data.frame()  %>% rename(
        UMAP1 = umap_1,
        UMAP2 = umap_2),
      adt@reductions$ref.uma@cell.embeddings %>% as.data.frame() %>%  rename(
        UMAP1 = refUMAP_1,
        UMAP2 = refUMAP_2
      )
    )
      df_umap <- df_umap %>% mutate(
        cell_name = rownames(df_umap), .before = 1)
#    df_umap <-  bind_rows(tibble(cell_name = Cells(adata_ref), UMAP1 = adata_ref$umap_1,
#   UMAP2 = adata_ref$umap_2), adt@reductions$ref.uma@cell.embeddings %>% as.tibble() %>%  rename(
#   UMAP1 = refUMAP_1,
#   UMAP2 = refUMAP_2) %>% mutate(
#   cell_name = rownames(adt@reductions$ref.uma@cell.embeddings)
# )
# )


  df_obs <- df_obs %>% left_join(df_umap, by = c("cell_name" = "cell_name"))
  rm(df_umap)
  # df_obs %>% head(2)
  # [out] predicted_count_[que].csv
  df_predicted_count <- df_obs %>% filter(dataset == sprintf("%s_%s", tissue_name, sp2)) %>% group_by(true_label, pre_label) %>% count() %>% pivot_wider(
    names_from = pre_label,
    values_from  = n
  )
  df_predicted_count %>% write_csv(file.path(resdir, sprintf("predicted_count_%s.csv", sp2)))

  # df_predicted_count #%>%head(2)

  # [out] ratio.csv
  # tissue,type,sp,name,is_right_sum,is_right_count,ratio
  df_ratio <- df_ratio <- df_obs %>% group_by(dataset) %>% summarise(
    tissue = tissue_name,
    type = "species",
    sp = str_split(dataset, "_")[[1]][2],
    name = "",
    is_right_sum = length(is_right),
    is_right_count = sum(is_right, na.rm = F),
    ratio = is_right_count / is_right_sum

  ) %>% select(-dataset)

  df_ratio %>% write_csv(file.path(resdir, "ratio.csv"))
  # [out] obs.csv
  df_obs %>% write_csv(file.path(resdir, "obs.csv"))



  # plot umap_dataset.png and umap_mapt.png
  p_fig <- file.path(resdir, "figs")
  if (!dir.exists(p_fig)) {
    dir.create(p_fig)
  }

  temp_theme <- theme(
    panel.background = element_blank(),
    axis.line = element_line(),
    legend.key = element_blank(),
    legend.title = element_blank(),
    # legend.text = element_text(size = 14)

  )
    
    
  p <- df_obs %>% ggplot(aes(x = UMAP1, y = UMAP2, color = dataset)) + geom_point(size = .5) + temp_theme
  print(p)
  ggsave(file.path(p_fig, "umap_dataset.png"), p, width = 110, height = 100, units = "mm")
  p <- df_obs %>% ggplot(aes(x = UMAP1, y = UMAP2, color = cell_type)) + geom_point(size = .5) + temp_theme
  print(p)
  ggsave(file.path(p_fig, "umap_umap.png"), p, width = 110, height = 100, units = "mm")

    # umap adata_ref Seurat_clusters
    p = DimPlot(adata_ref, group.by = c(key_celltype), reduction = "umap")
    print(p)
    ggsave(file.path(p_fig, sprintf("umap_ref_%s.png",key_celltype)), p, width = 110, height = 100, units = "mm")
    p = DimPlot(adata_ref, group.by = c("Seurat_clusters"), reduction = "umap")
    print(p)
    ggsave(file.path(p_fig, "umap_ref_Seurat_clusters.png"), p, width = 110, height = 100, units = "mm")
        
    # ElbowPlot
    p =   ElbowPlot(adata_ref, ndims = 50)
    ggsave(file.path(p_fig, "ElbowPlot_ref.png"), p, width = 110, height = 100, units = "mm")


    # calculate_confusion_matrix and F1-score
    cm = calculate_more_with_confusion_matrix(
    calculate_confusion_matrix(
        filter(df_obs,dataset == paste(tissue_name, sp2, sep = "_"))$true_label,
        filter(df_obs,dataset == paste(tissue_name, sp2, sep = "_"))$pre_label
    )
)
    cat('[confusion_matrix] head\n')
    print(cm %>% head)

    return(list(
        weighted_F1  = calculate_F1Score_with_confusion_matrix(cm,average = "weighted")
    )
          )
    
}


# ### run_Seurat

# In[12]:


run_Seurat <- function(
    path_adata1,
    path_adata2, path_varmap,
    key_class1,
    key_class2,
    sp1,
    sp2,
    tissue_name,
    dims,
    resolution,
    refdata,
    aligned = FALSE,
    resdir_tag = "",
    resdir = "/public/workspace/licanchengup/download/test/test_result",
    is_1v1 = TRUE,key_cell_name = "cell_name"
    ) {
  time_start <- as.numeric(Sys.time())
  ## before run --------------------------------------------------
  resdir <-  file.path(resdir, sprintf("%s_%s-corss-%s;%s", tissue_name, sp1, sp2, resdir_tag))
  # whether finish
  if (file.exists(file.path(resdir, "finish"))) {
    cat(sprintf("[has finish] %s\n", resdir))
    return()
  } else {
    cat(sprintf("[start] %s\n", resdir))
  }
    
  if (!dir.exists(resdir)) {
    dir.create(resdir,recursive=TRUE)
  }
  key_celltype <- ""
  if (key_class1 == key_class2) {
    key_celltype <- key_class1
  } else {
    stop(sprintf("key_class1, key_class2 is not equal\n %s != %s", key_class1, key_class2))
  }


# load ref and que
adata_ref <- load_Seuratobj(path_adata1, return_matrix = TRUE, add_obs = FALSE)
adata_que <- load_Seuratobj(path_adata2, return_matrix = TRUE, add_obs = FALSE)

#----------------------------------------
## homology one2one for adata_que
#----------------------------------------
n_homology_noe2one_find <- 0
n_homology_noe2one_use <- 0

if (is_1v1) {

df_varmap <- read_csv(path_varmap, na = "")[, 1:3]
colnames(df_varmap) <- c("gn_ref", "gn_que", "homology_type")
# 去除na与重复项
df_varmap <- df_varmap %>% filter(!is.na(gn_ref), !is.na(gn_que)) %>% distinct()
keep= (df_varmap %>% transmute(
gn_ref_is_unique = gn_ref %in% filter(df_varmap %>% group_by(gn_ref) %>% count() ,n == 1)$gn_ref,
gn_que_is_unique = gn_que %in% filter(df_varmap %>% group_by(gn_que) %>% count() ,n == 1)$gn_que,
keep = gn_ref_is_unique & gn_que_is_unique
))$keep
df_varmap = df_varmap[keep,] %>% filter(homology_type == 'ortholog_one2one')
df_varmap <- tibble(
gn_ref = rownames(adata_ref)
) %>% left_join(df_varmap, by = "gn_ref")

n_homology_noe2one_find <- df_varmap %>% filter(!is.na(gn_que)) %>% nrow()
cat(sprintf("[homology one2one]find %d genes\n", n_homology_noe2one_find))
# 没有配对的，使用ref的原名字
# # ref_SNORD14E对上了que_SNORD14
# # ref_SNORD14没对上，用原名字则导致que存在重复名字，故给没对上的加个前缀
# # SNORD14	SNORD14	NA
# # SNORD14E	SNORD14	ortholog_one2one
df_varmap <- df_varmap %>% mutate(
gn_que = ifelse(is.na(gn_que), paste("not_o2o", gn_ref, sep = "_"), gn_que)
)

if (!all(rownames(adata_ref) == df_varmap$gn_ref)) {
stop("df_varmap$gn_ref not equal to rownames(adata)")
}
if (df_varmap$gn_ref %>% duplicated %>% any | df_varmap$gn_que %>% duplicated %>% any) {
stop("df_varmap$gn_ref or df_varmap$gn_que is duplicated")
}
#--------------------
# homology one2one gene name convert
rownames(adata_ref)  <- df_varmap$gn_que
#--------------------

n_homology_noe2one_use <- intersect(
rownames(adata_ref),
rownames(adata_que)) %>% length()
cat(sprintf("[homology one2one]use %d genes\n", n_homology_noe2one_use))

}
# add obs if it is exists
adata_ref <- load_Seuratobj_add_obs(path_adata1, CreateSeuratObject(adata_ref),key_cell_name =key_cell_name )
adata_que <- load_Seuratobj_add_obs(path_adata2, CreateSeuratObject(adata_que),key_cell_name =key_cell_name )




  # [out] group_counts_unalign.csv
  df_group_counts <- bind_rows(
    # ref
    tibble(
      dataset = paste(tissue_name, sp1, sep = "_"),
      cell_type = adata_ref@meta.data[[key_celltype]]
    ),
    # que
    tibble(
      dataset = paste(tissue_name, sp2, sep = "_"),
      cell_type = adata_que@meta.data[[key_celltype]]
  )) %>% group_by(dataset, cell_type) %>% count() %>% pivot_wider(
    names_from = "dataset", values_from = "n"
  )
  df_group_counts %>% write_csv(file.path(resdir, "group_counts_unalign.csv"))
  # process aligend
  if (aligned) {
    inter_celltype <- intersect(adata_ref@meta.data[[key_celltype]], adata_que@meta.data[[key_celltype]])
    inter_celltype

    adata_ref@meta.data <- adata_ref@meta.data %>% mutate(
      in_inter_celltype__ = (.data[[key_celltype]] %in% inter_celltype)

    )
    adata_que@meta.data <- adata_que@meta.data %>% mutate(
      in_inter_celltype__ = (.data[[key_celltype]] %in% inter_celltype)

    )
    adata_ref <- subset(adata_ref, in_inter_celltype__)
    adata_que <- subset(adata_que, in_inter_celltype__)
  }

  df_group_counts <- bind_rows(
    # ref
    tibble(
      dataset = paste(tissue_name, sp1, sep = "_"),
      cell_type = adata_ref@meta.data[[key_celltype]]
    ),
    # que
    tibble(
      dataset = paste(tissue_name, sp2, sep = "_"),
      cell_type = adata_que@meta.data[[key_celltype]]
  )) %>% group_by(dataset, cell_type) %>% count() %>% pivot_wider(
    names_from = "dataset", values_from = "n"
  )
  df_group_counts %>% write_csv(file.path(resdir, "group_counts.csv"))

  time_before <- as.numeric(Sys.time())
  ## run --------------------------------------------------
  # for ref
  # NormalizeData FindVariableFeatures ScaleData RunPCA
  # ElbowPlot 决定 后续的dims
  adata_ref <- Map_Seurat_normalize(adata_ref)
  

  # for ref
  # FindNeighbors FindCluster RunUMAP(return.model = TRUE)
  # RunUMAP返回umap model，后续的Running UMAP projection 需要umap model
  # 调节dims, resolution
  adata_ref <- Map_Seurat_cluster(adata_ref, dims = dims, resolution = resolution, key_celltype = key_celltype)
  adata_ref

  # [out] obs_ref.csv obs_que.csv
  # obs_ref 中包含了 Seurat_clusters
  adata_ref@meta.data %>% write.csv(file.path(resdir, "obs_ref.csv"))
  adata_que@meta.data %>% write.csv(file.path(resdir, "obs_que.csv"))
  # for que
  # NormalizeData FindVariableFeatures
  adata_que <- Map_Seurat_normalize(adata_que, run_scale = FALSE, run_pca = FALSE)

  # for ref and que
  intersect_features <- intersect(Features(adata_ref), Features(adata_que))
  cat(sprintf("[intersect features] is %d\n", intersect_features   %>% length()
  ))

  if (length(intersect_features) <= 20) {
    cat(intersect_features, "\n")
  } else (
    cat("[intersect_features][top 20]\n", intersect_features[1:20], "\n")
  )


  # FindTransferAnchors MapQuery
  adt <- Map_Seurat_mapquery(adata_ref, adata_que, dims = dims, refdata = refdata)
  adt
  adt@meta.data %>% head(2)

  DimPlot(adt,
    reduction = "ref.umap", group.by = key_celltype, label = FALSE, label.size = 3,
    repel = TRUE
  )
  DimPlot(adt,
    reduction = "ref.umap", group.by = "predicted..", label = FALSE, label.size = 3,
    repel = TRUE
  )

  time_run <- as.numeric(Sys.time())

  res_after = precess_after_Seurat(resdir, adata_ref, adt, adata_que, key_celltype, tissue_name, sp1, sp2)

  time_end <- as.numeric(Sys.time())
  write((sprintf(
    "[start] %f
[finish before run]\t%f
[patameter][path_varmap]\t%s
[parameter][n_homology_noe2one_find]\t%d
[parameter][n_homology_noe2one_use]\t%d
[parameter][intersect_features_n]\t%d
[out][weighted_F1]\t%f
[finish run]\t%f
[end] %f", 
      time_start, time_before, path_varmap,
      n_homology_noe2one_find, n_homology_noe2one_use,
      intersect_features %>% length(),
      res_after[['weighted_F1']],time_run, time_end)),file.path(resdir, "finish"))
  cat(sprintf("[has finish] %s\n", resdir))
return(list(
adata_ref=adata_ref, 
adata_que=adata_que, 
adt=adt

))
}


# ### Map_Seurat_example

# In[13]:


Map_Seurat_example = function(Map_Seurat_example_index = NULL){
Map_Seurat_example_1 = "
p_src <- '.'
p_root <- './RA_h-corss-m;Seurat;AMP-Phase-1-map-GSE145286;CL_cell_subtype1'
p_root
path_adata1 <- file.path(p_src, 'AMP-Phase-1_human_fibroblast') # human
path_adata2 <- file.path(p_src, 'GSE145286_mouse_fibroblast') # mouse

# load ref and que --------------------------------------------------
## load ref
adata_ref <- load_Seuratobj(path_adata1)
adata_que <- load_Seuratobj(path_adata2)

# map  ------------------------------------------------------------
# ref -> human
# que -> mouse

# for ref
# NormalizeData FindVariableFeatures ScaleData RunPCA
# ElbowPlot 决定 后续的dims
adata_ref <- Map_Seurat_normalize(adata_ref)

# for ref
# FindNeighbors FindCluster RunUMAP(return.model = TRUE)
# RunUMAP返回umap model，后续的Running UMAP projection 需要umap model
# 调节dims, resolution

adata_ref <- Map_Seurat_cluster(adata_ref, dims = 1:10, resolution = 0.1, key_celltype = 'CL_cell_subtype1')
adata_ref

# for que
# NormalizeData FindVariableFeatures
adata_que <- Map_Seurat_normalize(adata_que, run_scale = FALSE, run_pca = FALSE)

# for ref and que
# FindTransferAnchors MapQuery
adata_res <- Map_Seurat_mapquery(adata_ref, adata_que, dims = 1:10, refdata = list('.' = 'CL_cell_subtype1'))
adata_res
adata_res@meta.data %>% head(2)

adata_res@meta.data %>% head(2)

DimPlot(adata_res,
  reduction = 'ref.umap', group.by = 'CL_cell_subtype1', label = FALSE, label.size = 3,
  repel = TRUE
)
DimPlot(adata_res,
  reduction = 'ref.umap', group.by = 'predicted..', label = FALSE, label.size = 3,
  repel = TRUE
)

"
    
if(Map_Seurat_example_index == 1){
    cat(Map_Seurat_example_1 )
}else{
    cat("Map_Seurat_example_index == 1
> [simply]
    use run_Seurat

Map_Seurat_example_index == 2
> [detail] 
    use Map_Seurat_normalize, Map_Seurat_cluster, Map_Seurat_mapquery
")
    
}

}


# # end
# 

# In[14]:


cat("
> function----------------------------------------
Seurat_to_mtx

> Map_Seurat function-----------------------------
Map_Seurat_normalize
Map_Seurat_cluster
Map_Seurat_mapquery
precess_after_Seurat
run_Seurat\t\t\t[simply]

> other-------------------------------------------
get_path_varmap
show_obj
savefig
Seurat_metadata_leftjoin
Seurat_gene_detect
")


# # Debug

# In[15]:


# Map_Seurat_example(1)


# In[16]:


# p_src <- '.'

# path_adata1 <- file.path(p_src, 'AMP-Phase-1_human_fibroblast') # human
# path_adata2 <- file.path(p_src, 'GSE145286_mouse_fibroblast') # mouse
# key_class1 = 'CL_cell_subtype1'
# key_class2 = 'CL_cell_subtype1'
# # key_celltype <- 'CL_cell_subtype1'
# sp1 <- 'h'
# sp2 <- 'm'
# tissue_name <- 'RA'
# aligned= FALSE
# refdata = list('.' = 'CL_cell_subtype1')
# resdir_tag <- paste('Seurat', 'test_F', sep = ';')
# resdir <- file.path('/public/workspace/licanchengup/link/disease/test_disease_2')


# In[17]:


# p = "/public/workspace/ruru_97/projects/data/homo/biomart/input/human_to_mouse_1v1.txt"
# p = "/public/workspace/ruru_97/projects/data/homo/biomart/input/human_to_mouse.txt"
# p = '/public/workspace/licanchengup/link/test/came_sample_data/gene_matches_mouse2human.csv'

