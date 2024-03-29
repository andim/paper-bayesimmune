import string, itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def label_axes(fig_or_axes, labels=string.ascii_uppercase,
               labelstyle=r'{\sf \textbf{%s}}',
               xy=(0.0, 1.0), **kwargs):
    """
    Walks through axes and labels each.
    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure or Axes to work on
    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.
    labelstyle : format string
    kwargs : to be passed to annotate (default: ha='left', va='top')
    """
    # re-use labels rather than stop labeling
    labels = itertools.cycle(labels)
    axes = fig_or_axes.axes if isinstance(fig_or_axes, plt.Figure) else fig_or_axes
    defkwargs = dict(ha='left', va='top') 
    defkwargs.update(kwargs)
    for ax, label in zip(axes, labels):
        xycoords = (ax.yaxis.label, 'axes fraction')
        ax.annotate(labelstyle % label, xy=xy, xycoords=xycoords, **defkwargs)

def despine(ax, spines=['top', 'right']):
    if spines == 'all':
        spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_visible(False)

class OffsetHandlerTuple(matplotlib.legend_handler.HandlerTuple):
    """
    Legend Handler for tuple plotting markers on top of each other
    """
    def __init__(self, **kwargs):
        matplotlib.legend_handler.HandlerTuple.__init__(self, **kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        nhandles = len(orig_handle)
        perside = (nhandles - 1) / 2
        offset = height / nhandles
        handler_map = legend.get_legend_handler_map()
        a_list = []
        for i, handle1 in enumerate(orig_handle):
            handler = legend.get_legend_handler(handler_map, handle1)
            _a_list = handler.create_artists(legend, handle1,
                                             xdescent,
                                             offset*i+ydescent-offset*perside,
                                             width, height,
                                             fontsize,
                                             trans)
            a_list.extend(_a_list)
        return a_list
