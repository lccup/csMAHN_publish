#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""utils.scanpy as ut.sc
single cell
"""


# In[2]:


from utils.general import Path, np, pd, plt, mpl, sns, json
from utils.general import show_dict_key, archive_gzip,handle_type_to_list
import scanpy as sc
import scipy
from collections.abc import Iterable

from PIL import Image
# 有的组织图像太大，取消Image的图像大小限制
Image.MAX_IMAGE_PIXELS = None


# # img

# In[ ]:


def reverse_img(
        img_arr,
        reverse_h=False,
        reverse_v=False,
        transpose=False):
    """旋转图片

Parameters
----------
reverse_h : 水平方向是否颠倒
reverse_v : 垂直方向是否颠倒
transpose : 是否旋转
    """
    def reverse_color_layer(
            arr,
            reverse_h=True,
            reverse_v=True,
            transpose=False):
        """对rgb颜色分量进行转置

Parameters
----------
arr : np.arrary 2d
    颜色分量,2维 数组
        """
        assert len(arr.shape) == 2
        res = None
        if reverse_h and reverse_v:
            res = arr[::-1, ::-1]
        elif reverse_v:
            res = arr[::-1, :]
        elif reverse_h:
            res = arr[:, ::-1]
        else:
            # reverse_h = False and reverse_v = False
            res = arr
        if transpose:
            res = res.transpose()
        return res

    def yield_color_layer(arr):
        """获取颜色分量

Parameters
----------
arr :
    图片数组, 3维 数组
    (M,N,3) or (M,N,4)
    r,g,b      r,g,b,a
        """
        assert len(arr.shape) == 3
        for i in range(arr.shape[2]):
            yield arr[:, :, i]

    def merge_color_layer(*args):
        return np.dstack(args)

    return merge_color_layer(*[reverse_color_layer(_,
                                                   reverse_h=reverse_h, reverse_v=reverse_v,
                                                   transpose=transpose)
                               for _ in yield_color_layer(img_arr)])


def show_reverse_img(img_arr):
    df_parameter = pd.MultiIndex.from_product(
        [[0, 1], [0, 1], [0, 1]],
        names='transpose,reverse_h,reverse_v'.split(','))\
        .to_frame().reset_index(drop=True)
    df_parameter.index = df_parameter.eval(
        'transpose*2 + reverse_h*4 + reverse_v')
    df_parameter = df_parameter.apply(lambda x: x > 0)

    fig, axs = plt.subplots(ncols=4, nrows=2)
    axs = np.ravel(axs)

    for i, row in df_parameter.iterrows():
        ax = axs[i]
        ax.imshow(reverse_img(img_arr, **row))
        ax.set_axis_off()
        ax.text(
            0,
            0,
            'transpose={transpose}\nreverse_h={reverse_h}\nreverse_v={reverse_v}' .format(
                **row),
            verticalalignment='top')


def reverse_spatial_img(adata, key_uns_spatial='spatial',
                        key_images=None,
                        reverse_h=True, reverse_v=True,
                        transpose=False):
    """转置切片

Parameters
----------
key_uns_spatial : str (default: 'spatial')
    用于指定样本操作
key_images : list,str (default: None)
    用于限制操作的图像,默认全部操作
"""
    assert 'spatial' in adata.uns.keys()

    dict_spatial = adata.uns['spatial'][key_uns_spatial]
    if key_images is None:
        key_images = dict_spatial['images'].keys()

    for k_img, img_arr in dict_spatial['images'].items():
        dict_spatial['images'][k_img] = reverse_img(img_arr,
                                                    reverse_h=reverse_h,
                                                    reverse_v=reverse_v,
                                                    transpose=transpose)
    return adata


# # show

# In[ ]:


def show(adata, show_adata=False, show_var=False, show_check_unique=False,
         show_key_images=False,
         show_X_gt_zero=False, n=2,):
    from IPython.display import display
    display(adata) if show_adata else None
    if show_check_unique:
        print("""[check unique]
\tobs.index\tvar.index
\t{}\t\t{}""".format(adata.obs.index.is_unique, adata.var.index.is_unique))
    display(adata.obs.head(n), adata.obs.shape)
    display(adata.var.head(n), adata.var.shape) if show_var else None
    display(adata.X[adata.X > 0][:10]) if show_X_gt_zero else None

    if show_key_images:
        display(pd.DataFrame(np.array(
            list(yield_key_uns_spatial_and_key_img(adata))),
            columns='key_uns_spatial,key_img'.split(',')).groupby(
            'key_uns_spatial,key_img'.split(',')
        ).agg({'key_img': 'count'}).loc[:, []])


def show_spatial(adata):
    if 'spatial' in adata.obsm.keys():
        print("# adata.obsm['spatial']".ljust(75, '-'))
        print(adata.obsm['spatial'][:3], '...\nshape: ',
              adata.obsm['spatial'].shape, '\n')
    else:
        print("[not exists] adata.obsm['spatial']")

    if 'spatial' in adata.uns.keys():
        print("# adata.uns['spatial']".ljust(75, '-'))
        show_dict_key(adata.uns['spatial'], "adata.uns['spatial']")
        for key_uns_spatial, dict_spatial in adata.uns['spatial'].items():
            show_dict_key(dict_spatial, key_uns_spatial)
            _k = 'images' if 'images' in dict_spatial.keys() else 'images_path'
            if _k in dict_spatial.keys():
                show_dict_key(
                    dict_spatial[_k], "{}['{}']".format(
                        key_uns_spatial, _k))
            else:
                print("[no imgs]")
    else:
        print("[not exists] adata.uns['spatial']")


# # get and yield

# In[ ]:


def get_key_uns_spatial(adata):
    assert 'spatial' in adata.uns.keys()
    return list(adata.uns['spatial'].keys())


def get_dict_spatial(adata, key_uns_spatial):
    return adata.uns['spatial'][key_uns_spatial]


def yield_dict_spatial(adata):
    for key_uns_spatial in get_key_uns_spatial(adata):
        yield get_dict_spatial(adata, key_uns_spatial)


def yield_key_uns_spatial_and_key_img(adata):
    for key_uns_spatial in get_key_uns_spatial(adata):
        dict_spatial = get_dict_spatial(adata, key_uns_spatial)
        _k = 'images' if dict_spatial.setdefault(
            'images', {}) else 'images_path'
        for k_img in dict_spatial.setdefault(_k, {}).keys():
            yield (key_uns_spatial, k_img)


def get_img(adata, key_uns_spatial='spatial', key_img='img'):
    """尝试从adata.uns['spatial'][key_uns_spatial]\
['images']['key_img']
"""
    return adata.uns['spatial'][key_uns_spatial]\
        .setdefault('images', {})\
        .setdefault(key_img, None)


def get_scalefactor(adata, key_uns_spatial='spatial', key_img='img',
                    key_scalefactors_format='tissue_{}_scalef',
                    default_value=1):
    """尝试从adata.uns['spatial'][key_uns_spatial]\
['scalefactors'][key_scalefactors_format.format(key_img)]
中获取scalefactor
若失败则返回default_value

Parameters
----------
default_value : int,float (default: 1)
"""
    return adata.uns['spatial'][key_uns_spatial]\
        .setdefault('scalefactors', {})\
        .setdefault(key_scalefactors_format.format(key_img), default_value)


def get_spot_size(adata, key_uns_spatial='spatial', default_value=1):
    """尝试从adata.uns['spatial'][key_uns_spatial]\
['scalefactors']['spot_diameter_fullres']
中获取scalefactor
若失败则返回default_value

Parameters
----------
default_value : int,float (default: 1)
"""
    return adata.uns['spatial'][key_uns_spatial]\
        .setdefault('scalefactors', {})\
        .setdefault('spot_diameter_fullres', default_value)

def get_obs_df(adata,keys,layer=None):
    keys = handle_type_to_list(keys)
    temp = [k for k in keys if not (k in adata.var_names or k in adata.obs_keys())]
    if len(temp) > 0:
        print("[Waring] field not in adata : {}".format(','.join(temp)))
    keys = [k for k in keys if k not in temp]
    return sc.get.obs_df(adata,keys,layer=layer)

def get_gene_mean(adata,genes,key_group,layer=None):
    genes = handle_type_to_list(genes,str)
    data = get_obs_df(adata,[key_group]+genes)
    return data.groupby(key_group).mean()
def get_gene_exp_pct(adata,genes,key_group,layer='counts'):
    genes = handle_type_to_list(genes,str)
    data = get_obs_df(adata,key_group).join(get_obs_df(adata,genes)>0)
    data = data.groupby(key_group).sum()/data.groupby(key_group).count()
    return data


# # I/O

# In[ ]:


def load_adata(p_dir, prefix=''):
    def _load_json(p):
        return json.loads(p.read_text())

    def load_h5ad_from_mtx(p_dir, prefix=''):
        p_dir = Path(p_dir)

        if p_dir.joinpath('{}matrix.mtx.gz'.format(prefix)).exists():
            archive_gzip(p_dir.joinpath('{}matrix.mtx.gz'.format(prefix)),
                         decompress=True, remove_source=False, show=False)

        assert p_dir.joinpath('{}matrix.mtx'.format(prefix)).exists(
        ), '[not exists] {}matrix.mtx or {}matrix.mtx.gz\n in {}'.format(prefix, p_dir)

        adata = sc.read_10x_mtx(p_dir, prefix=prefix)

        p_dir.joinpath(
            '{}matrix.mtx'.format(prefix)).unlink() if p_dir.joinpath(
            '{}matrix.mtx.gz'.format(prefix)).exists() else None

        # obs.csv
        if p_dir.joinpath('{}obs.csv'.format(prefix)).exists():
            adata.obs = pd.read_csv(
                p_dir.joinpath(
                    '{}obs.csv'.format(prefix)),
                index_col=0)
        else:
            print('[not exists]{}obs.csv\nin {}'.format(prefix, p_dir))
        return adata

    p_dir = Path(p_dir)
    adata = None
    if p_dir.match("*.h5ad"):
        adata = sc.read_h5ad(p_dir)
    elif p_dir.is_dir() and (
        p_dir.joinpath('{}matrix.mtx'.format(prefix)).exists() or p_dir\
            .joinpath('{}matrix.mtx.gz'.format(prefix)).exists()
    ):
        adata = load_h5ad_from_mtx(p_dir, prefix)
    else:
        raise Exception("[can not load adata] {}".format(p_dir))

    # [load] spatial info: adata.obsm and adata.uns['spatial']
    # adata.obsm
    if p_dir.joinpath('{}obsm'.format(prefix)).exists():
        for p_obsm in p_dir.joinpath('{}obsm'.format(prefix)).iterdir():
            if not p_obsm.match('*csv'):
                continue
            adata.obsm[p_obsm.stem] = pd.read_csv(p_obsm).to_numpy()
    # adata.uns['spatial']
    if p_dir.joinpath('{}uns/spatial'.format(prefix)).exists():
        adata.uns['spatial'] = {}
        for p_uns_spatial in p_dir.joinpath('{}uns/spatial'.format(prefix)
                                            ).iterdir():

            dict_spatial = {'images': {}}
            for img in p_uns_spatial.joinpath('images').iterdir():
                dict_spatial['images'][img.stem] = plt.imread(img)

            for p in p_uns_spatial.iterdir():
                if p.match('*.json'):
                    dict_spatial[p.stem] = _load_json(p)

            adata.uns['spatial'][p_uns_spatial.stem] = dict_spatial

    return adata


def load_spatial_images(
        adata,
        key_uns_spatial=None,
        key_images_path='images_path',
        key_images=None,
        kw_reverse_img={'reverse_h': False,
                        'reverse_v': False,
                        'transpose': False},
        update_images=False):
    """加载切片图像
从dict_spatial['key_images_path']加载
到dict_spatial['images']中

Parameters
----------
key_uns_spatial : str (default: None)
    用于指定样本名称,默认全部导入
key_images : list,str (default: None)
    用于限制导入的图像,默认全部导入
"""
    def load_spatial_images_one(
            adata,
            key_uns_spatial,
            key_images_path='images_path',
            key_images=None,
            kw_reverse_img={},
            update_images=False):
        assert key_uns_spatial in adata.uns['spatial'].keys()
        dict_spatial = adata.uns['spatial'][key_uns_spatial]

        assert key_images_path in dict_spatial.keys()
        assert dict_spatial[key_images_path], "[Error] no img path"

        if 'images' not in dict_spatial.keys():
            dict_spatial['images'] = {}

        if key_images is None:
            key_images = list(dict_spatial[key_images_path].keys())
        if isinstance(key_images, str):
            key_images = [key_images]

        for k_img_path, img_path in dict_spatial[key_images_path].items():
            if k_img_path not in key_images:
                continue
            if update_images or (
                    k_img_path not in dict_spatial['images'].keys()):

                dict_spatial['images'][k_img_path] = reverse_img(
                    mpl.image.imread(img_path), **kw_reverse_img)

        adata.uns['spatial'][key_uns_spatial] = dict_spatial
        return adata

    assert 'spatial' in adata.uns.keys()
    if key_uns_spatial is None:
        key_uns_spatial = adata.uns['spatial'].keys()
    elif isinstance(key_uns_spatial, str):
        key_uns_spatial = [key_uns_spatial]

    for k in key_uns_spatial:
        adata = load_spatial_images_one(
            adata,
            k,
            key_images_path,
            key_images,
            kw_reverse_img,
            update_images)
    return adata


def load_obsm_from_obs(adata, keys_obs, key_obsm, drop_obs=False):
    if isinstance(keys_obs, str):
        keys_obs = [keys_obs]
    adata.obsm[key_obsm] = adata.obs.loc[:, keys_obs].to_numpy()
    adata.obs = adata.obs.drop(
        columns=keys_obs) if drop_obs else adata.obs
    return adata


def load_obsm_spatial(adata, keys_pixel='pixel_x,pixel_y'.split(','),
                      drop_obs=False):
    return load_obsm_from_obs(adata, keys_pixel, 'spatial', drop_obs)


def load_obsm_UMAP(
        adata, keys_umap='UMAP1,UMAP2'.split(','),
        drop_obs=False):
    return load_obsm_from_obs(adata, keys_umap, 'X_umap', drop_obs)


def load_uns_spatial(adata, key_uns_spatial='spatial',
                     path_imgs=None,
                     path_jsons=None):
    """加载adata.uns['spatial'][key_uns_spatial]
Parameters
----------
key_uns_spatial : str (default: 'spatial')
    pass
path_imgs : dict, str, Path (default: None)
    None is not allowed
    if provide str or Path, the key of img will be 'img'
"""
    assert path_imgs is not None, '[Error] no imgs'
    if isinstance(path_imgs, (str, Path)):
        path_imgs = {'img': Path(path_imgs)}

    dict_spatial = {k: {} for k in 'images_path'.split(',')}
    dict_spatial['images_path'] = path_imgs
    dict_spatial.update(
        {k: json.loads(Path(v).read_text())
         for k, v in path_jsons.items()})

    if 'spatial' not in adata.uns.keys():
        adata.uns['spatial'] = {}
    adata.uns['spatial'][key_uns_spatial] = dict_spatial
    return adata


def load_uns_monocle2(adata, p_dir):
    """加载adata.uns['monocle2']
Parameters
----------
p_dir : str|Path
    load_adata的路径，将此路径下的 ./uns/monocle2 导入
"""
    p_monocle2 = Path(p_dir).joinpath("uns", "monocle2")
    assert p_monocle2.exists(), "[not exists] {}".format(p_monocle2)
    adata.uns['monocle2'] = {
        'edge_df': pd.read_csv(
            p_monocle2.joinpath("edge_df.csv")), 'data_df': pd.read_csv(
            p_monocle2.joinpath("data_df.csv"), index_col=0)}
    if p_monocle2.joinpath("branch_points").exists():
        adata.uns['monocle2']['branch_points'] = np.array(
            [i for i in p_monocle2.joinpath("branch_points")
             .read_text().split('\n')
             if len(i) > 0])
    adata.obs = adata.obs.join(adata.uns['monocle2']['data_df']
                               .loc[:, 'Pseudotime,State'.split(',')])
    return adata


def save_as_mtx(adata, p_dir, layer='counts', prefix='', as_int=True,
                key_images_path='images_path'):
    """
    将adata对象保存为matrix.mtx,barcodes.tsv,genes.tsv
    尝试保存obs.csv
    尝试保存空间信息,adata.obsm and adata.uns['spatial']
    p_dir ： 输出路径
    as_int : 是否将矩阵转换int类型
        default True
"""
    def _save_img(arr, p):
        # 存为npz和npy都极其巨大
        # 图片压缩技术是真的强
        im = Image.fromarray(arr)
        im.save(p)

    def _save_json(data, p):
        p.write_text(json.dumps(data))

    assert adata.obs.index.is_unique, '[Error] obs index is not unique'
    assert adata.var.index.is_unique, '[Error] var index is not unique'

    p_dir = Path(p_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    # [out] genes.tsv
    adata.var["gene_names"] = adata.var_names.to_numpy()
    if "gene_ids" not in adata.var_keys():
        adata.var["gene_ids"] = adata.var["gene_names"]
    df_genes = adata.var.loc[:, ["gene_ids", "gene_names"]]
    df_genes.to_csv(
        p_dir.joinpath("{}genes.tsv".format(prefix)),
        header=False,
        index=False,
        sep="\t",
    )
    # [out] barcodes.tsv obs.csv
    adata.obs.loc[:, []].to_csv(
        p_dir.joinpath("{}barcodes.tsv".format(prefix)),
        header=False,
        index=True,
        sep="\t",
    )

    if len(adata.obs_keys()) > 0:
        adata.obs.to_csv(
            p_dir.joinpath("{}obs.csv".format(prefix)), index=True
        )

    # [out] matrix.mtx
    data = adata.layers[layer] if layer in adata.layers.keys() else adata.X
    data = scipy.sparse.csr_matrix(data)
    if as_int:
        data = data.astype(int)
    nonzero_index = [i[:10] for i in data.nonzero()]
    print(
        "frist 10 matrix nonzero elements:\n",
        data[nonzero_index[0], nonzero_index[1]],
    )
    scipy.io.mmwrite(
        p_dir.joinpath("{}matrix.mtx".format(prefix)), data.getH()
    )

    # [save] spatial info: adata.obsm and adata.uns['spatial']
    if 'spatial' in adata.uns.keys() and 'spatial' in adata.obsm.keys():
        print('[save] spatial info')
        # [save] adata.obsm
        p_dir.joinpath(
            '{}obsm'.format(prefix)).mkdir(
            parents=True,
            exist_ok=True)
        for k_obsm in adata.obsm.keys():
            pd.DataFrame(
                adata.obsm[k_obsm],
                index=adata.obs.index,
                columns=[
                    '{}{}'.format(
                        k_obsm,
                        _+
                        1) for _ in range(
                        adata.obsm[k_obsm].shape[1])]) .to_csv(
                    p_dir.joinpath(
                        '{}obsm/{}.csv'.format(
                            prefix,
                            k_obsm)),
                index=False)

        # [save] adata.uns['spatial']
        for k_spatial, dict_spatial in adata.uns['spatial'].items():
            p_dir.joinpath('{}uns/spatial/{}/images'.format(prefix,
                           k_spatial)) .mkdir(parents=True, exist_ok=True)
            # img 仅保存dict_spatial[key_images_path]中没有路径的img

            for k_img, img in dict_spatial.setdefault(
                    'images', {}).items():
                if k_img in dict_spatial.setdefault(
                        key_images_path, {}).keys():
                    pass
                else:
                    _save_img(img, p_dir.joinpath(
                        '{}uns/spatial/{}/images/{}.jpg'
                        .format(prefix, k_spatial, k_img)))

            if dict_spatial.setdefault(key_images_path, None):
                _save_json({_k: str(Path(_v).absolute())
                            for _k, _v in dict_spatial[key_images_path].items()},
                           p_dir.joinpath('{}uns/spatial/{}/{}.json'
                                          .format(prefix, k_spatial, key_images_path)))
            # scalefactors
            if dict_spatial.setdefault('scalefactors', None):
                _save_json(dict_spatial['scalefactors'],
                           p_dir.joinpath('{}uns/spatial/{}/scalefactors.json'
                                          .format(prefix, k_spatial)))
            # metadata
            if dict_spatial.setdefault('metadata', None):
                _save_json(dict_spatial['metadata'],
                           p_dir.joinpath('{}uns/spatial/{}/metadata.json'
                                          .format(prefix, k_spatial)))

    archive_gzip(p_dir.joinpath('matrix.mtx'), show=False)
    print("[out] {}".format(p_dir))


# # Standard Process

# In[ ]:


def qc(adata,drop_total_counts = True):
    adata.var["mt"] = adata.var_names.str.match('^MT-', case=False)
    adata.var["ribo"] = adata.var_names.str.match('^RP[SL]', case=False)
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]", case=False)
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars='mt,ribo,hb'.split(','),
        percent_top=[],
        log1p=False,
        inplace=True)

    if drop_total_counts:
        adata.obs = adata.obs.loc[:,~adata.obs.columns.str.match('^total_counts_')] 
    return adata


def normalize_and_log1p(
        adata,
        save_layers_counts=False,
        save_layers_normalize=False):
    if save_layers_counts:
        adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata)

    if save_layers_normalize:
        adata.layers["normalize"] = adata.X.copy()

    sc.pp.log1p(adata, base=np.e)
    return adata


def standard_process(
    adata, kv_hvg={
        'n_top_genes': 2000, 'batch_key': None}, kv_neighbors={
            'n_neighbors': 15, 'n_pcs': 15}, kv_umap={}, kv_leiden={
                'resolution': .5}):
    """
单细胞标准流程

+ 将原始count矩阵存于adata.layers["counts"]
+ 标准化
    + sc.pp.normalize_total
    + sc.pp.log1p
+ hvgs
    + highly_variable_genes
+ 降维 PCA
    + sc.tl.pca
        + sc.pl.pca_variance_ratio
+ 降维 UMAP
    + sc.pp.neighbors
    + sc.tl.umap
    + sc.tl.leiden
        + sc.pl.umap
"""
    adata = adata.copy()
    addata = normalize_and_log1p(adata)
    sc.pp.highly_variable_genes(adata, **kv_hvg)
    sc.tl.pca(adata, n_comps=50)
    sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)
    sc.pp.neighbors(adata, **kv_neighbors)
    sc.tl.umap(adata, **kv_umap)
    sc.tl.leiden(adata, **kv_leiden)
    sc.pl.umap(adata, color=["leiden"])
    return adata


# # other

# In[ ]:


def subset_adata(adata, *args):
    """

Parameters
----------
adata : anndata
    AnnData object
*args : pair condition


Examples
--------
subset_adata(adata, '_batch', 'A')
subset_adata(adata, '_batch', ['A','B'])
subset_adata(adata,
        '_batch', 'A,B'.split(','),  # 1st pair condition
        'cell_type','T cell'         # 2nd pair condition
        )

        """
    def _process_values(values):
        if isinstance(values, Iterable):
            if isinstance(values, str):
                values = [values]
        else:
            values = [values]
        return values
    assert len(
        args) % 2 == 0, '[Error][{}] length of args must be 2*n'.format(len(args))

    for key, values in zip(args[0::2], args[1::2]):
        values = _process_values(values)
        adata = adata[adata.obs[key].isin(values), :]
    return adata

