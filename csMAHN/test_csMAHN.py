import sys
import os
import time
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/public/workspace/ruru_97/projects/schgnn/csMAHN')
import utils.preprocess as pp
from utils.utility import *
from utils.train import Trainer
from utils.plot import plot_umap


def test_scMAHN(
        reverse=False,
        aligned=True,
        tissue="pancreas",
        gse_ids=["GSE84113", "GSE84113"],
        species=('human', 'mouse'),
        dsnames=('human', 'mouse'),
        path_data='/public/workspace/ruru_97/projects/data',
        resdir='/public/workspace/ruru_97/projects/schgnn/result',
        homo_method='biomart',
        key_class='cell_type',
        n_hvgs=2000,
        n_degs=50,

        seed = 123,
        stages=[200, 200, 200],
        nfeats=64,  # enbedding size #128
        hidden=64,  # 128
        input_drop=0.2,
        att_drop=0.2,
        residual=True,

        threshold=0.9, # 0.8
        lr=0.01,  # lr = 0.01
        weight_decay=0.001,
        patience=100,
        enhance_gama=10,
        simi_gama=0.1 ):
    # reverse reference to query, query to reference.
    if reverse:
        gse_ids = gse_ids[::-1]
        species = species[::-1]
        dsnames = dsnames[::-1]
    seed_all(seed)
    path_homo = f'{path_data}/homo/{homo_method}/input/{species[0]}_to_{species[1]}.txt'
    path_specie_1 = f'{path_data}/ByTissue/{tissue}/{gse_ids[0]}/input/{dsnames[0]}.h5ad'
    path_specie_2 = f'{path_data}/ByTissue/{tissue}/{gse_ids[1]}/input/{dsnames[1]}.h5ad'

    # make file to save
    time_tag = make_nowtime_tag()
    curdir = f'{resdir}/{tissue}/{tissue}-{gse_ids[0]}_{dsnames[0]}-{gse_ids[1]}_{dsnames[1]}-{time_tag}'
    model_dir = os.path.join(curdir, 'model_')
    figdir = os.path.join(curdir, 'fig_')
    os.mkdir(curdir)
    os.mkdir(figdir)
    os.mkdir(model_dir)
    checkpt_file = model_dir + "/mutistages"
    print(checkpt_file)

    for i in range(len(stages)):
        res_dir = os.path.join(curdir, f'res_{i}')
        os.mkdir(res_dir)
    homo = pd.read_csv(path_homo)
    adata_species_1 = sc.read_h5ad(path_specie_1)
    adata_species_2 = sc.read_h5ad(path_specie_2)

    if aligned:
        adata_species_1, adata_species_2 = pp.aligned_type([adata_species_1, adata_species_2], 'cell_type')

    print(
        f'Task: refernece:{gse_ids[0]}_{dsnames[0]} {adata_species_1.shape[0]} cells x {adata_species_1.shape[1]} gene -> query:{gse_ids[1]}_{dsnames[1]} {adata_species_2.shape[0]} cells x {adata_species_2.shape[1]} gene in {tissue}')

    start = time.time()
    # knn时间较长
    adatas, features_genes, nodes_genes, scnets, one2one, n2n = pp.process_for_graph([adata_species_1, adata_species_2],
                                                                                     homo,
                                                                                     key_class,
                                                                                     'leiden',
                                                                                     n_hvgs=n_hvgs,
                                                                                     n_degs=n_degs)
    g, inter_net, one2one_gene_nodes_net, cell_label, n_classes, list_idx = pp.make_graph(adatas,
                                                                                          aligned,
                                                                                          key_class,
                                                                                          features_genes,
                                                                                          nodes_genes,
                                                                                          scnets,
                                                                                          one2one,
                                                                                          n2n,
                                                                                          has_mnn=True,
                                                                                          seed=seed)
    end = time.time()
    # 包括预处理时间
    print('Times preprocess for graph:{:.2f}'.format(end - start))

    
    trainer = Trainer(adatas,
                      g,
                      inter_net,
                      list_idx,
                      cell_label,
                      n_classes,
                      threshold=threshold,
                      key_class=key_class)
    trainer.train(curdir=curdir, 
                  checkpt_file=checkpt_file,

                  nfeats=nfeats,
                  hidden=hidden,
                  enhance_gama=enhance_gama,
                  simi_gama=simi_gama)


    plot_umap(trainer.embedding_hidden, adatas, dsnames, figdir)