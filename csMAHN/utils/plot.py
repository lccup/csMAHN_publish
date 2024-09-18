import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# def plot_umap(embedding_hidden, adatas, dsnames, figdir,
#               dpi_save=200, n_neighbors=15, metric='cosine', use_rep='X', key_class='cell_type'):
#     adt = sc.AnnData(embedding_hidden.detach().numpy())
#     adt.obs['cell_type'] = adatas[0].obs[key_class].tolist() + adatas[1].obs[key_class].tolist()
#     adt.obs['dataset'] = [dsnames[0]] * adatas[0].shape[0] + [dsnames[1]] * adatas[1].shape[0]
#     sc.set_figure_params(dpi_save=200)
#     sc.settings.figdir = figdir
#     sc.pp.neighbors(adt, n_neighbors=n_neighbors, metric=metric, use_rep='X')
#     sc.tl.umap(adt)

#     sc.pl.umap(adt, color='dataset', save='_dataset.png')
#     sc.pl.umap(adt, color='cell_type', save='_umap.png')


def plot_umap(adt, key_class,
              figdir, dpi_save=200, n_neighbors=15, metric='cosine', use_rep='X'):

    sc.set_figure_params(dpi_save=200)
    sc.settings.figdir = figdir
    sc.pp.neighbors(adt, n_neighbors=n_neighbors, metric=metric, use_rep='X')
    sc.tl.umap(adt)

    sc.pl.umap(adt, color='dataset', save='_dataset.png')
    sc.pl.umap(adt, color=key_class, save='_umap.png')

def plot_confusion(data):
    sns.set_theme()
    f, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(data,linewidths=.5, ax=ax)