# -*- encoding: utf-8 -*-
'''
@File    :   method.py
@Time    :   2023/05/30 11:29:52
@Author  :   UynajGI
@Version :   0.4.3
@Contact :   suquan12148@outlook.com
@License :   (MIT)Copyright 2023
'''

# here put the import lib
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import HTMLWriter
from .algorithm.Metropolis import _rename
from typing import Dict
import pickle
import uuid

__all__ = [
    'mean', 'std', 'var', 'norm', 'diff', 'cv', 'u4', 'getcolumn', 'curve',
    'scatter', 'param_plot', 'param_scatter', 'imshow', 'animate', 'to_msdt',
    'read_msdt', 'setup_uid'
]


def mean(algo, uid: str, column: str, t0: int = 0, n: int = 1) -> float:
    column = _rename(column)
    return np.mean(algo.data.loc[uid][column][t0:]**n)


def std(algo, uid: str, column: str, t0: int = 0) -> float:
    column = _rename(column)
    return np.std(algo.data.loc[uid][column][t0:])


def var(algo, uid: str, column: str, t0: int = 0) -> float:
    column = _rename(column)
    return np.var(algo.data.loc[uid][column][t0:])


def norm(algo, uid: str, column: str, t0: int = 0, ord: int = 2) -> float:
    column = _rename(column)
    return np.linalg.norm(algo.data.loc[uid][column][t0:], ord=ord)


def diff(algo, uid: str, column: str, t0: int = 0, n: int = 1) -> np.array:
    column = _rename(column)
    return np.diff(algo.data.loc[uid][column][t0:], n)


def cv(algo, uid: str, column: str, t0: int = 0) -> float:
    column = _rename(column)
    return algo.std(uid, column, t0) / algo.mean(uid, column, t0)


def u4(algo, uid: str, t0: int = 0) -> float:
    return 1 - algo.mean(uid, "magnetization", t0=t0, n=4) / (
        3 * algo.mean(uid, "magnetization", t0=t0, n=2)**2)


def getcolumn(algo, uid: str, column: str, t0: int = 0) -> np.array:
    column = _rename(column)
    return algo.data.loc[uid][column][t0:]


def curve(algo, uid, column, t0: int = 0) -> None:
    """
    Draw a curve.
    """
    data = algo.data
    column = _rename(column)
    array = data.loc[uid][column][t0:]
    index = data.loc[uid].index
    plt.plot(index, array)


def scatter(algo,
            uid,
            column,
            s=None,
            c=None,
            t0: int = 0,
            marker=None,
            cmap=None,
            norm=None,
            vmin=None,
            vmax=None,
            alpha=None,
            linewidths=None,
            *,
            edgecolors=None,
            plotnonfinite=False,
            data=None,
            **kwargs) -> None:

    data = algo.data
    column = _rename(column)
    array = data.loc[uid][column][t0:]
    index = data.loc[uid].index
    plt.scatter(index,
                array,
                s=s,
                c=c,
                marker=marker,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                linewidths=linewidths,
                edgecolors=edgecolors,
                plotnonfinite=plotnonfinite,
                data=data,
                **kwargs)


def param_plot(algo,
               uid_dict: Dict[str, np.array],
               column: str,
               per: bool = True) -> None:
    """
    Draw a parametric plot.
    """
    column = _rename(column)
    x = []
    y = []
    if isinstance(uid_dict, dict):
        param_name = list(uid_dict.keys())[1]
        for i in range(len(list(uid_dict.values())[0])):
            uid = list(uid_dict.values())[0][i]
            param = list(uid_dict.values())[1][i]
            if uid not in algo.data.index.get_level_values("uid").values:
                raise ValueError("Invalid uid.")
            x.append(param)
            y.append(algo.mean(uid, column))
    else:
        raise ValueError("Invalid uid_dict.")
    if per:
        plt.plot(x, y / algo.model.N, label=param_name)
    else:
        plt.plot(x, y, label=param_name)


def param_scatter(algo,
                  uid_dict: Dict[str, np.array],
                  column: str,
                  per: bool = True) -> None:
    """
    Draw a parametric scatter.
    """
    column = _rename(column)
    x = []
    y = []
    if isinstance(uid_dict, dict):
        param_name = list(uid_dict.keys())[1]
        for i in range(len(list(uid_dict.values())[0])):
            uid = list(uid_dict.values())[0][i]
            param = list(uid_dict.values())[1][i]
            if uid not in algo.data.index.get_level_values("uid").values:
                raise ValueError("Invalid uid.")
            x.append(param)
            y.append(algo.mean(uid, column))
    else:
        raise ValueError("Invalid uid_dict.")
    if per:
        plt.scatter(x, y / algo.model.N, label=param_name)
    else:
        plt.scatter(x, y, label=param_name)


def imshow(algo,
           uid: str,
           iter: int,
           cmap: str = 'gray',
           norm=None,
           aspect=None,
           interpolation=None,
           alpha=None,
           vmin=None,
           vmax=None,
           origin=None,
           extent=None,
           interpolation_stage=None,
           filternorm=True,
           filterrad=4.0,
           resample=None,
           url=None,
           data=None,
           **kwargs) -> None:
    """
    Draw a inshow.
    """
    spin = algo.data.loc[(uid, iter), "spin"]
    plt.imshow(spin,
               cmap=cmap,
               norm=norm,
               aspect=aspect,
               interpolation=interpolation,
               alpha=alpha,
               vmin=vmin,
               vmax=vmax,
               origin=origin,
               extent=extent,
               interpolation_stage=interpolation_stage,
               filternorm=filternorm,
               filterrad=filterrad,
               resample=resample,
               url=url,
               data=data,
               **kwargs)
    plt.axis('off')
    plt.axis('equal')


def animate(algo, uid: str, save: bool = False, savePath: str = None) -> None:
    """
    Animate the spin.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    spin_lst = algo.data.loc[uid, 'spin'].tolist()

    def init():
        ax.imshow(spin_lst[0], cmap='gray')
        ax.axis('off')
        return ax

    def update(iter):
        ax.clear()
        ax.imshow(spin_lst[iter], cmap='gray')
        ax.set_title('iter: {}'.format(iter))
        ax.axis('off')
        return ax

    ani = animation.FuncAnimation(fig=fig,
                                  func=update,
                                  init_func=init,
                                  frames=range(len(spin_lst)))
    if save:
        mywriter = HTMLWriter(fps=60)
        if savePath is None:
            if not os.path.exists(uid):
                os.mkdir(uid)
            os.chdir(uid)
        else:
            if not os.path.exists(savePath):
                os.mkdir(savePath)
            os.chdir(savePath)
        ani.save('myAnimation.html', writer=mywriter)
        plt.close()
        os.chdir('..')


def to_msdt(algo, path: str = '.msdt'):
    """
    Convert the data to msdt.
    """
    # 检查 path 最后是否有'.msdt'后缀
    if path[-5:] != '.msdt':
        path += '.msdt'
    # 检查是否存在文件
    if os.path.exists(path):
        # 存在覆盖
        os.remove(path)
    model = algo._rowmodel
    data = algo.data
    param_list = algo.param_list
    name = algo.name
    savedata = {
        'model': model,
        'data': data,
        'param_list': param_list,
        'name': name
    }
    open(path, 'wb').write(pickle.dumps(savedata))


def read_msdt(path: str = None):
    """
    Read the data from msdt.
    """
    # 检查是否存在文件
    if not os.path.exists(path):
        raise FileNotFoundError("File not found.")
    return pickle.loads(open(path, 'rb').read())


def setup_uid(algo, uid):
    if uid is None:
        uid = (uuid.uuid1()).hex
    else:
        if algo.data.index.get_level_values("uid").values:
            if uid not in algo.data.index.get_level_values("uid").values:
                algo._reset_model()
            else:
                algo.model.set_spin(algo.data.loc[uid].loc[
                    algo.data.loc[uid].index.max()].spin)
    return uid
