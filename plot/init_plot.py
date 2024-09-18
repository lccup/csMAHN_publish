#!/usr/bin/env python
# coding: utf-8

# In[33]:


import sys
from pathlib import Path
p_temp = str(Path('~/link/res_publish').expanduser())
None if p_temp in sys.path else sys.path.append(p_temp)
del p_temp


# In[2]:


import utils as ut
from utils.general import *
pl = ut.pl
sc = ut.sc.sc


# In[3]:


from func import *

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

cm = mpl.colormaps['YlOrRd']
cm_2 = mpl.colormaps['magma']


# In[5]:


fontdict_default = dict(fontsize=6)
fontdict_axes_title = dict(fontsize=8)


# # plot para
# 

# In[6]:


ppara_data = {}
ppara_adata = {}
ppara_cmap = {}
ppara_func = {}


# # EnrichmentAnalysis
# 
# 借由脚本`Rscript EnrichmentAnalysis.r` (R 富集分析器)实现在python通过shell调用指定conda环境中的R
# 
# 进行EnrichAnalysis

# In[7]:


def EnrichmentAnalysis_run_with_r(
        tag,
        geneset_label,
        gene_string,
        separator=',',
        env='publish'):
    command = "{}Rscript EnrichmentAnalysis.r 'data/EnrichmentAnalysis/{}' '{}' '{}' {}".format(
        '~/apps/miniconda3/envs/{}/bin/'.format(env) if env else '',
        tag, geneset_label, gene_string, separator

    )
    os.system(command)


def EnrichmentAnalysis_show_genesets(p=Path(
        '~/link/res_publish/plot/data/EnrichmentAnalysis').expanduser()):
    gs_info = pd.read_csv(p.joinpath('geneset/geneset_info.csv'))
    display(gs_info.loc[:, 'label,version'.split(',')])


def EnrichmentAnalysis_find_genesets_path(geneset_label, p=Path(
        '~/link/res_publish/plot/data/EnrichmentAnalysis').expanduser()):
    gs_info = pd.read_csv(p.joinpath('geneset/geneset_info.csv'))
    gs_info['path'] = gs_info['path'].apply(
        lambda x: p.joinpath('geneset', x))
    res = gs_info.query("label == '{}'".format(geneset_label))
    if res.shape[0] == 1:
        res = res['path'].to_list()[0]
    else:
        raise Exception("[Error] find {} itme with geneset_label='{}'"
                        .format(res.shape[0], geneset_label))
    return res


def EnrichmentAnalysis_get_res_df(
        p=p_plot.joinpath('data/EnrichmentAnalysis')):
    assert p.exists(), '[not exists] {}'.format(p)
    df = pd.DataFrame({'dir': p.iterdir()})
    df = df[~df['dir'].apply(lambda x: x.match('*/.ipynb_checkpoints'))]
    df.index = df['dir'].apply(lambda x: x.name)
    df['p_gene'] = df['dir'].apply(lambda x: x.joinpath('gene_input.txt'))
    df['p_table'] = df['dir'].apply(
        lambda x: x.joinpath('Enrich_table.csv'))
    df['p_table_all'] = df['dir'].apply(
        lambda x: x.joinpath('Enrich_table_all.csv'))
    return df


def EnrichmentAnalysis_get_adj(row, all=False):
    return pd.read_csv(
        row['p_table{}'.format('_all' if all else '')],
        index_col=0).loc[:, ['p.adjust']].rename(columns={
            'p.adjust': row.name
        })


# # dotplot

# In[ ]:


def dotplot_marker(adata, genes, key_group, ax, order=None, zscore=True,
                   layer=None, layer_counts='counts', kw_bubble=None,transpose = True):
    """注意: 使用a4p时，1*1 的格子画一个点
    """
    kw_bubble = update_dict({}, kw_bubble)
    if isinstance(genes,(np.ndarray,pd.Series)):
        genes = list(genes)
    
    df_mean = ut.sc.get_gene_mean(adata, key_group=key_group, genes=genes, layer=layer)
    df_pct = ut.sc.get_gene_exp_pct(adata, key_group=key_group, genes=genes, layer=layer_counts)
    
    if order:
        order = handle_type_to_list(order,str)
        df_mean = ut.df.get(df_mean, order)
        df_pct = ut.df.get(df_pct, order)
    
    if zscore:
        df_mean = ut.sc.scipy.stats.zscore(df_mean, axis=0)
    if transpose:
        df_mean = df_mean.transpose()
        df_pct = df_pct.transpose()
    cbar = pl.bubble(df_pct*200, ax, df_mean, fontdict_xtick=dict(rotation=90), **kw_bubble)
    return cbar

def dotplot_marker_legend(ax, draw_cbar=True,fig=None, ax_cbar=None, cbar=None):
    """
    # 大小固定
    ax = a4p.add_ax(5, 1, 1, 3, rc=update_dict(pl.rc_blank, {'ytick.labelright': True, 'xtick.labelbottom': True}))
    ax_cbar = a4p.add_ax(5+1, 1, 1, 3, rc=pl.rc_blank)
"""

    temp = pd.DataFrame({'percentage': [.25, .5, .75]})
    temp.index = temp['percentage'].apply(lambda x: '{:.0f}%'.format(x*100)).to_numpy()
    pl.bubble(temp.transpose()*200, ax, 'black')
    if draw_cbar:
        assert not cbar is None and  not fig is None
        ax_cbar = ax if ax_cbar is None else ax_cbar
        fig.colorbar(cbar, ax=ax_cbar)

def dotplot_max_prob_median(data,ax,order_row = None,order_col=None,kw_bubble=None):
    def max_prob_median(data):
        data = data.groupby('true_label,pre_label'.split(','))['max_prob'].median()\
            .reset_index().pivot(index = 'true_label',columns= 'pre_label',values='max_prob')\
            .fillna(0)
        return data

    """
    注意：pl.bubble 所绘制的矩阵图是经过.transpose()的
    故 order_col 与 order_row 分别对应 data_prob 和 data_ratio 的
        index 和 column
"""
    kw_bubble = update_dict(dict(fontdict_xtick=dict(rotation=90)),kw_bubble)
    data_prob = max_prob_median(data).transpose()
    data_ratio = ut.df.matrix_classify(data,'true_label','pre_label')\
        .pipe(pl.bar_count_to_ratio)
    if order_row is None and order_col is None:
        order_row = np.unique(np.sort(np.concatenate([data_prob.index,data_prob.columns])))
        order_col = order_row
        
    order_row = np.sort(data_prob.columns) if order_row is None else order_row
    order_col = np.sort(data_prob.index) if order_col is None else order_col
    
    data_prob = ut.df.get(data_prob,order_col,order_row)
    data_ratio = ut.df.get(data_ratio,order_col,order_row)
    
    cbar = pl.bubble(data_ratio*200, ax, data_prob,**kw_bubble)
    return cbar


# # sc_pl_show_genes
# 
# 借由scanpy 进行绘图, 并做了些许调整
# 
# 但是可恶的错误提示仍然未能避免，气

# In[8]:


# funcs_select_axes = {
#     '0_colorbar': lambda fig,
#     marker: [
#         i for i in fig.get_axes() if i.get_title() == 'z-score of\nmean expression'],
#     '1_box': lambda fig,
#     marker: [
#         i for i in fig.get_axes() if list(
#             marker.keys()) == [
#             i1.get_text() for i1 in i.get_yticklabels()]],
# }


# def sc_pl_stacked_violin(adata, marker, group_by, categories_order=None,
#                          ax=None, del_yticks=False, var_group_rotation=0,
#                          funcs_select_axes=funcs_select_axes):
#     if categories_order is None:
#         categories_order = np.sort(adata.obs[group_by].unique())
#     else:
#         categories_order = [_ for _ in categories_order
#                             if _ in adata.obs[group_by].unique()]

#     sc.pl.stacked_violin(
#         adata, marker, group_by,
#         var_group_rotation=var_group_rotation,
#         colorbar_title="z-score of\nmean expression",
#         vmax=2.5, vmin=-2.5, cmap=cm,  # cmap="Blues",
#         # layer="scaled",
#         dendrogram=False, swap_axes=False,
#         ax=ax, show=False,  # return_fig=True
#     )
#     fig = ax.figure

#     for i in funcs_select_axes['0_colorbar'](fig, marker):
#         fig.delaxes(i)
#     for i in funcs_select_axes['1_box'](fig, marker):
#         i.tick_params(length=0, width=0)
#         i.set_frame_on(False)
#         i.set_yticks([], []) if del_yticks else None

#     for _patches in np.concatenate([i.patches for i in fig.get_axes()
#                                     if list(marker.keys()) == [
#         i2.get_text() for i2 in i.get_children()
#             if isinstance(i2, mpl.text.Text) and len(i2.get_text()) > 0]
#     ]):
#         _vertices = _patches.get_path().vertices
#         _vertices[:, 1] = np.where(
#             _vertices[:, 1] == 0.6, 0.1, _vertices[:, 1])
#         _patches.set_path(
#             mpl.path.Path(
#                 _vertices,
#                 _patches.get_path().codes))
#         _patches.set(linewidth=.75, linestyle='-', color='grey')
#     fig.draw_without_rendering()


# def sc_pl_matrixplot(adata, marker, group_by, categories_order=None,
#                      ax=None, del_yticks=False, var_group_rotation=0,
#                      funcs_select_axes=funcs_select_axes):
#     sc.pl.matrixplot(
#         adata, marker, group_by,
#         var_group_rotation=var_group_rotation,
#         colorbar_title="z-score of\nmean expression",
#         vmax=2.5, vmin=-2.5, cmap=cm,  # cmap="Blues",
#         layer="scaled",
#         dendrogram=False, swap_axes=False,
#         ax=ax, show=False,  # return_fig=True
#     )
#     fig = ax.figure

#     for i in funcs_select_axes['0_colorbar'](fig, marker):
#         fig.delaxes(i)
#     for i in funcs_select_axes['1_box'](fig, marker):
#         i.tick_params(length=0, width=0)
#         i.set_frame_on(False)
#         i.set_yticks([], []) if del_yticks else None

#     for _patches in np.concatenate([i.patches for i in fig.get_axes()
#                                     if list(marker.keys()) == [
#         i2.get_text() for i2 in i.get_children()
#             if isinstance(i2, mpl.text.Text) and len(i2.get_text()) > 0]
#     ]):
#         _vertices = _patches.get_path().vertices
#         _vertices[:, 1] = np.where(
#             _vertices[:, 1] == 0.6, 0.1, _vertices[:, 1])
#         _patches.set_path(
#             mpl.path.Path(
#                 _vertices,
#                 _patches.get_path().codes))
#         _patches.set(linewidth=2, linestyle='-', color='grey')
#     fig.draw_without_rendering()


# map_func_sc_pl_show_genes = {
#     'stacked_violin': sc_pl_stacked_violin,
#     'matrixplot': sc_pl_matrixplot,
# }


# def sc_pl_show_genes(key, adata, marker, group_by, categories_order=None,
#                      ax=None, del_yticks=False, var_group_rotation=0,
#                      funcs_select_axes=funcs_select_axes):
#     """
#     key:
#         stacked_violin
#         matrixplot
#     """

#     map_func_sc_pl_show_genes[key](
#         adata, marker=marker, group_by=group_by,
#         categories_order=categories_order,
#         var_group_rotation=var_group_rotation,
#         ax=ax, del_yticks=del_yticks,
#         funcs_select_axes=funcs_select_axes
#     )


# In[9]:


ppara_data['key_scpl_show_genes'] = 'stacked_violin'


# # ppara_func
# 
# ## heatmap_gene
# 
# 借助sns实现的,仿照sc.pl.matrixplot

# In[ ]:


# with Block("ppara_func['heatmap_gene_get_marker_and_df_plot']"):
#     def _func(adata, group_by, marker, layer='scaled'):
#         marker = pd.concat([pd.DataFrame({
#             'cell_type': k,
#             'gene': list(vs)
#         }) for k, vs in marker.items()])
#         marker.index = np.arange(marker.shape[0])
#         marker['line_start'] = marker.groupby('cell_type').cumcount() == 0
#         marker['line_end'] = [False] + marker['line_start'].to_list()[1:]
#         df_plot = sc.get.obs_df(
#             adata,
#             [group_by] +
#             list(
#                 marker['gene'].unique()),
#             layer=layer)
#         df_plot = ut.df.group_agg(
#             df_plot, [group_by], {
#                 g: 'mean' for g in marker['gene']},reindex=False,recolumn=False)
#         df_plot = df_plot.loc[:, marker['gene']]
#         df_plot.columns = pd.MultiIndex.from_frame(
#             marker.loc[:, 'cell_type,gene'.split(',')])
#         return marker, df_plot

#     ppara_func['heatmap_gene_get_marker_and_df_plot'] = _func

# with Block("ppara_func['heatmap_gene_process_multi_marker_and_df_plot']"):
#     def _func(list_data):
#         def add_col_gap(df, key, values):
#             df[key] = values
#             return df

#         df_plot = pd.concat([add_col_gap(_[1], ('gap', 'gap_{}'.format(
#             _i)), np.nan) for _i, (_) in enumerate(list_data)], axis=1).copy()

#         df_temp = df_plot.columns.to_frame()
#         df_temp.index = np.arange(df_temp.shape[0])
#         df_temp['group'] = df_temp['gene'].str.extract(
#             'gap_(\\d+)', expand=False)
#         df_temp['group'] = df_temp['group'].bfill()

#         df_marker = df_marker = [_[0].assign(**{'group': str(_i)})
#                                  for _i, (_) in enumerate(list_data)]
#         df_marker = pd.concat(df_marker, axis=0)
#         df_marker = pd.merge(
#             df_temp,
#             df_marker,
#             on='group,cell_type,gene'.split(','),
#             how='left')
#         df_marker['line_start'] = df_marker['line_start'].mask(
#             df_marker['line_start'].isna(), False)
#         df_marker['line_end'] = df_marker['line_end'].mask(
#             df_marker['gene'].str.match('gap_\\d+'), True)
#         df_marker['cell_type_mask'] = df_marker['cell_type'].mask(
#             df_marker['cell_type'].str.match('^gap$') &
#             df_marker['gene'].str.match('^gap_\\d+$'), '')
#         df_marker['gene_mask'] = df_marker['gene'].mask(
#             df_marker['cell_type'].str.match('^gap$') &
#             df_marker['gene'].str.match('^gap_\\d+$'), '')
#         return df_marker, df_plot

#     ppara_func['heatmap_gene_process_multi_marker_and_df_plot'] = _func

# with Block("ppara_func['heatmap_gene']"):
#     def _func(
#             df_plot,
#             df_marker,
#             ax,
#             zscore=True,
#             cmap=cm,
#             cbar=False,
#             line_h=.25,
#             kv_line={
#                 'linestyle': '-',
#                 'linewidth': 1,
#                 'color': 'grey',
#                 'dash_joinstyle': 'miter',
#                 'dash_capstyle': 'butt'},
#             kv_heatmap={
#                 'vmax': 2.5,
#                 'vmin': -2.5},
#             **kvargs):
#         """
#            kvargs:
#                del_yticks: default False
#                kv_line_text: default fontdict_default

# """
#         df_marker.index = np.arange(df_marker.shape[0])
#         df_temp = df_plot.columns.to_frame().copy()
#         df_temp.index = np.arange(df_temp.shape[0])
#         assert (df_marker.loc[:, 'cell_type,gene'.split(',')] == df_temp).all(
#         ).all(), """[Error] df_plot.columns is not equal to
#  df_marker.loc[:,'cell_type,gene'.split(',')]"""

#         # if df_marker['gene'][-1].startswith('gap_'):
#         if not df_marker['gene_mask'].to_numpy()[-1]:
#             df_plot = df_plot.iloc[:, :-1].copy()
#         if zscore:
#             df_plot = ut.sc.scipy.stats.zscore(df_plot, axis=0, nan_policy='omit')
#         sns.heatmap(df_plot, square=True, cbar=cbar,
#                     cmap=cmap, ax=ax, **kv_heatmap)
#         ax.set_xlabel(''), ax.set_ylabel('')
#         if kvargs.setdefault('del_yticks', False):
#             ax.set_yticks([], [], 
#                           rotation=0,
#                           **fontdict_default)
#         else:
#             _texts = df_plot.index
#             ax.set_yticks(
#                 np.arange(
#                     df_plot.shape[0])+.5,
#                 df_plot.index,
#                 rotation=0,
#                 **fontdict_default)
#         _texts = df_marker[df_marker['gene_mask'].str.len()
#                            > 0]['gene_mask']
#         ax.set_xticks(_texts.index + .5, _texts, rotation=90,
#                       **fontdict_default)
#         ax.set_ylim(min(ax.get_ylim()), max(ax.get_ylim())+1)

#         # plot line
#         for text, s, e in zip(
#                 df_marker.query('line_start')['cell_type'],
#                 df_marker.query('line_start').index,
#                 df_marker.query('line_end').index.to_list() + [df_marker.index.size]):
#             h_s, h = df_plot.shape[0] + .25, line_h
#             s = s+.25
#             e = e-.25
#             # text = 'mast cell'
#             ax.step([s, s, e, e], [h_s, h_s+h, h_s+h, h_s], **kv_line)
#             _fontdict = fontdict_default.copy()
#             _fontdict.update({'horizontalalignment': 'center',
#                               'multialignment': 'center',
#                               'verticalalignment': 'bottom'
#                               })

#             ax.text((s+e)/2, h_s+h+.2, text,
#                     **kvargs.setdefault('kv_line_text', _fontdict))

#     ppara_func['heatmap_gene'] = _func


# ## other

# In[ ]:


with Block("ppara_func['ut_sc_umap_gene']"):
    def func(a4p,x,y,adata,gene,draw_cbar=False,kw_umap_gene=None,kw_cbar=None,return_cbar=False):
        """
Parameters
----------
kw_umap_gene : dict
    dict(
        size=.5,
        cmap='Purples'
    )

kw_cbar : dict
    dict(
        format='{x:.0f}',
        aspect=10
    )

Returns
-------
    ax      return_cbar=False
    ax,cbar return_cbar=True
"""
        kw_umap_gene = update_dict(dict(size=.5,cmap='Purples'),kw_umap_gene)
        kw_cbar = update_dict(dict(format='{x:.0f}',aspect=10),kw_cbar)
        ax = a4p.add_ax(x,y,3,3)
        cbar = ut.sc.pl.umap_gene(adata,gene,
                ax,draw_cbar=False,**kw_umap_gene)
        ax.set_title(gene)
        ax.set_rasterized(True)
        
        if draw_cbar:
            cbar = a4p.fig.colorbar(cbar,ax=a4p.add_ax(x+2.2,y+1.5,1,1.5,
                    rc=pl.rc_blank),**kw_cbar)
            cbar.ax.tick_params('both', width=.5, length=1.5, color='black')

        if return_cbar:
            return ax,cbar
        return ax

    ppara_func['ut_sc_umap_gene'] = func
    del func
with Block("ppara_func['rgba_to_hex']"):
    def _func(rgba):
        if len(rgba) == 4:
            r, g, b, a = rgba
        if len(rgba) == 3:
            r, g, b = rgba
        if max(r,g,b) < 1:
            r,g,b  = int(r*255),int(g*255),int(b*255)
        
        r_hex = hex(r)[2:].zfill(2)
        g_hex = hex(g)[2:].zfill(2)
        b_hex = hex(b)[2:].zfill(2)
        return "#{}{}{}".format(r_hex, g_hex, b_hex)

    ppara_func['rgba_to_hex'] = _func
with Block("ppara_func['get_color_from_cm']"):
    def _func(value, key_cm):
        return ppara_func['rgba_to_hex'](
            mpl.colormaps[key_cm](value))

    ppara_func['get_color_from_cm'] = _func

    del _func
with Block("ppara_func['legend_umap']"):
    def _func(a4p,x,y,fontdict = None):
        ax = a4p.add_ax(x,y,rc=pl.tl_rc(pl.rc_frame,pl.rc_tl_notick))
        ax.set_ylabel('UMAP2',fontdict = update_dict(dict(rotation=90),fontdict))
        ax.set_xlabel('UMAP1',fontdict = update_dict(dict(rotation=0),fontdict))
    ppara_func['legend_umap'] = _func
    del _func
with Block("df_varmap_query_exists"):
    def _func(df_varmap, list_gn_ref=[], list_gn_que=[], model="both"):
        df_varmap = df_varmap.copy()
        df_varmap["gn_ref_exists"] = df_varmap["gn_ref"].isin(list_gn_ref)
        df_varmap["gn_que_exists"] = df_varmap["gn_que"].isin(list_gn_que)
        if model == "both":
            df_varmap = df_varmap.query("gn_ref_exists & gn_que_exists")
        elif model == "ref":
            df_varmap = df_varmap.query("gn_ref_exists")
        elif model == "que":
            df_varmap = df_varmap.query("gn_que_exists")
        else:
            raise Exception("[Error] model must be one of both, ref, que")
        df_varmap = df_varmap.drop(columns="gn_ref_exists,gn_que_exists".split(","))
        return df_varmap
    ppara_func["df_varmap_query_exists"] = _func
    del _func


# # 多重检验 multiple_test

# In[11]:


def multiple_test(data, key_groupby, key_value, test_pairs,
                  test_func, test_func_kwarg={}, fd_method='bh'):
    """多重检验
    """
    def _get_values(data, key_groupby, value, key_value):
        return data.query(
            "{} == '{}'".format(
                key_groupby,
                value))[key_value].to_numpy()

    def _group_false_discovery_control(
            df, key_groupby='value_x', fd_method='bh'):

        df_list = []
        for g in df[key_groupby].unique():
            pass

            _temp = res.query("{} == '{}'".format(key_groupby, g)).copy()
            _temp['padj'] = ut.sc.scipy.stats.false_discovery_control(
                _temp['pvalue'], method=fd_method)
            df_list.append(_temp)
        return pd.concat(df_list)

    # 处理 test_pairs
    if isinstance(test_pairs[0], str):
        test_pairs[0] = [test_pairs[0]]
    if len(test_pairs[0]) == 1:
        test_pairs[0] = [test_pairs[0][0]
                         for _ in range(len(test_pairs[1]))]
    test_pairs[0] = list(test_pairs[0])
    test_pairs[1] = list(test_pairs[1])
    _temp = np.setdiff1d(
        np.unique(
            np.ravel(test_pairs)),
        data[key_groupby].unique())
    assert _temp.size == 0, '[Error] not all element of test_pairs are in data[key_gourpby]\n\t{}'\
        .format(','.join(_temp))
    # 多重假设检验并校正pvalue
    res = pd.MultiIndex.from_arrays(test_pairs).to_frame(
        name='value_x,value_y'.split(','))
    res = res.drop_duplicates(
        'value_x,value_y'.split(',')).query("value_x != value_y")

    res['mean_x'] = res.apply(lambda row: np.mean(_get_values(
        data, key_groupby, row['value_x'], key_value)), axis=1)
    res['mean_y'] = res.apply(lambda row: np.mean(_get_values(
        data, key_groupby, row['value_y'], key_value)), axis=1)
    res['mean_diff'] = res.eval('mean_y - mean_x')
    res['percent_mean_diff'] = res.eval('mean_diff/mean_x * 100')

    res['test_res'] = res.apply(
        lambda row: test_func(
            x=_get_values(
                data,
                key_groupby,
                row['value_x'],
                key_value),
            y=_get_values(
                data,
                key_groupby,
                row['value_y'],
                key_value),
            **test_func_kwarg),
        axis=1)
    res['statistic'] = res['test_res'].apply(lambda x: x.statistic)
    res['pvalue'] = res['test_res'].apply(lambda x: x.pvalue)
    # 以value_x分组对pvalue进行校正
    res = _group_false_discovery_control(res, 'value_x', fd_method)
    return res.drop(columns=['test_res'])

