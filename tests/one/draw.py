# -*- encoding: utf-8 -*-
"""
@File    :   draw.py
@Time    :   2023/05/16 20:16:41
@Author  :   UynajGI
@Version :   beta0.0.1
@Contact :   suquan12148@outlook.com
@License :   (MIT)Copyright 2023
"""

# here put the import lib
import matplotlib.pyplot as plt

# from matplotlib import animation
from . import analysis


class Plot:

    def __init__(self, algorithm):

        self.algorithm = algorithm

    def curve(self, uid, column):
        """
        Draw a curve.
        """
        data = self.algorithm.iter_data
        column = analysis._rename(column)
        array = data.loc[uid][column]
        index = data.loc[uid].index
        plt.plot(index, array)

    def scatter(self,
                uid,
                column,
                s=None,
                c=None,
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
                **kwargs):

        data = self.algorithm.iter_data
        column = analysis._rename(column)
        array = data.loc[uid][column]
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

    def param_plot(self, uid_dict, column):
        """
        Draw a parametric plot.
        """
        column = analysis._rename(column)
        x = []
        y = []
        param_name = list(uid_dict.keys())[1]
        for uid in uid_dict:
            if uid not in self.iter_data.index.get_level_values("uid").values:
                raise ValueError("Invalid uid.")
            x.append(analysis.mean(self.algorithm, uid, param_name))
            y.append(analysis.mean(self.algorithm, uid, column))
        plt.plot(x, y)

    def param_scatter(self, uid_dict, column):
        """
        Draw a parametric scatter.
        """
        column = analysis._rename(column)
        x = []
        y = []
        param_name = list(uid_dict.keys())[1]
        for uid in uid_dict:
            if uid not in self.iter_data.index.get_level_values("uid").values:
                raise ValueError("Invalid uid.")
            x.append(analysis.mean(self.algorithm, uid, param_name))
            y.append(analysis.mean(self.algorithm, uid, column))
        plt.scatter(x, y)

    def imshow(self,
               uid,
               iter,
               cmap='gray',
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
               **kwargs):
        """
        Draw a inshow.
        """
        spin = self.algorithm.iter_data.loc[(uid, iter), "spin"]
        pic = plt.imshow(spin,
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
        return pic


class Animation(Plot):

    def __init__(self, algorithm):
        if self.algorithm.model.type != "ising":
            raise ValueError("Only Ising model can be animated.")
        if self.algorithm.model.dimension != 2:
            raise ValueError("Only 2D Ising model can be animated.")
        super().__init__(algorithm)

    # TODO[0.4.0]: Add animation function.