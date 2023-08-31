import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import gridspec
from horizon.problem import Problem
from horizon.variables import InputVariable, Variable, RecedingVariable, RecedingInputVariable
from horizon.functions import Function, RecedingFunction

import math
import numpy as np
import casadi as cs
import random

import matplotlib
matplotlib.use("Qt5Agg")

# TODO:
#   label: how to reduce it
#   actually opts must be set from outside

def create_grid(n_plots, title, n_rows_max, opts=None):
    if opts is None:
        opts = {}

    cols = n_plots if n_plots < n_rows_max else n_rows_max
    rows = int(math.ceil(n_plots / cols))

    gs = gridspec.GridSpec(rows, cols, **opts)

    fig = plt.figure(layout='constrained')
    fig.suptitle(title)

    return fig, gs

class Plotter():
    def __init__(self, prb: Problem, solution):
        self.prb = prb
        self.solution = solution
        self.axes = dict()

        # gs.hspace = 0.3

        # w = 7195
        # h = 3841

        # fig = plt.figure(frameon=True)
        # fig.set_size_inches(fig_size[0], fig_size[1])
        # fig.tight_layout()

    def setSolution(self, solution):
        self.solution = solution

    def plot(self, item, dim=None, args=None, ax=None):

        plt_name = f'{item.getName()}'
        if dim is not None:
            plt_name += f'_{dim}'

        if plt_name in self.axes:
            ax = self.axes[plt_name]

        if ax is None:
            plt.figure(frameon=True)
            plt.title(plt_name)
            ax = plt.gca()

        # store ax in memory
        self.axes[plt_name] = ax

        if args is None:
            args = {}


        val = self.__get_val(item, dim)

        if isinstance(item, (InputVariable, RecedingInputVariable)):
            args['drawstyle'] = 'steps-pre'

        return self.__plot_element(val, args, self.axes[plt_name])

    def __get_val(self, item, dim):

        if isinstance(item, (Variable, RecedingVariable)):
            val = self.solution[item.getName()]
        elif isinstance(item, (Function, RecedingFunction)):
            val = self.prb.evalFun(item, self.solution)
        else:
            raise ValueError('item not recognized')

        var_dim_select = np.array(range(val.shape[0]))

        if dim is not None:
            if np.setxor1d(var_dim_select, np.array(dim)).size == 0:
                raise Exception('Wrong selected dimension.')
            else:
                var_dim_select = dim

        return val[var_dim_select, :]

    def __plot_element(self, val, args, ax):


        if not ax.lines:
                ax.plot(val.T, **args)
        else:
                for index, line in zip(range(val.shape[1]), ax.lines):
                    line.set_data(range(val.shape[1]), val[index, :])
                    ax.draw_artist(line)

                ax.get_figure().canvas.restore_region(ax.get_figure().canvas.copy_from_bbox(ax.bbox))
                # redraw just the points
                # fill in the axes rectangle
                ax.get_figure().canvas.blit(ax.bbox)
                ax.get_figure().canvas.flush_events()

        return ax

    # def __plot_bounds(self, ax, abstract_var, dim, args=None):
    #
    #     val, var_dim_select = self.__get_val(abstract_var, dim)
    #     nodes_var = val.shape[1]
    #
    #     lb, ub = abstract_var.getBounds()
    #
    #     for dim in var_dim_select:
    #         ax.plot(np.array(range(nodes_var)), lb[dim, range(nodes_var)], *args)
    #         ax.plot(np.array(range(nodes_var)), ub[dim, range(nodes_var)], *args)
    #L


class PlotterHorizon:
    def __init__(self, prb: Problem, solution=None, opts=None, logger=None):

        self.solution = solution
        self.prb = prb
        self.logger = logger
        self.opts = opts

    def setSolution(self, solution):
        self.solution = solution

    def _plotVar(self, val, ax, abstract_var, markers, show_bounds, legend, dim):
        var_dim_select = set(range(val.shape[0]))
        nodes_var = val.shape[1]
        if dim is not None:
            if not set(dim).issubset(var_dim_select):
                raise Exception('Wrong selected dimension.')
            else:
                var_dim_select = dim

        if nodes_var == 1:
            markers = True

        baseline = None
        legend_list = list()
        if isinstance(abstract_var, InputVariable):
            for i in var_dim_select: # get i-th dimension

                r = random.random()
                b = random.random()
                g = random.random()
                color = (r, g, b)

                for j in range(nodes_var-1):
                    # ax.plot(np.array(range(val.shape[1])), val[i, :], linewidth=0.1, color=color)
                    # ax.plot(range(val.shape[1])[j:j + 2], [val[i, j]] * 2, color=color)
                    ax.step(np.array(range(nodes_var)), val[i, range(nodes_var)], linewidth=0.1, color=color, label='_nolegend_')

                    if show_bounds:
                        lb, ub = abstract_var.getBounds()

                        if markers:
                            ax.plot(range(nodes_var), lb[i, range(nodes_var)], marker="x", markersize=3, linestyle='dotted',linewidth=1, color=color)
                            ax.plot(range(nodes_var), ub[i, range(nodes_var)], marker="x", markersize=3, linestyle='dotted',linewidth=1, color=color)
                        else:
                            ax.plot(range(nodes_var), lb[i, range(nodes_var)], linestyle='dotted')
                            ax.plot(range(nodes_var), ub[i, range(nodes_var)], linestyle='dotted')

                if legend:
                    legend_list.append(f'{abstract_var.getName()}_{i}')
                    if show_bounds:
                        legend_list.append(f'{abstract_var.getName()}_{i}_lb')
                        legend_list.append(f'{abstract_var.getName()}_{i}_ub')
        else:
            for i in var_dim_select:
                if markers:
                    baseline, = ax.plot(range(nodes_var), val[i, :], marker="o", markersize=2)

                else:
                    baseline, = ax.plot(range(nodes_var), val[i, :])

                if show_bounds:
                    lb, ub = abstract_var.getBounds()
                    lb_mat = np.reshape(lb, (abstract_var.getDim(), len(abstract_var.getNodes())), order='F')
                    ub_mat = np.reshape(ub, (abstract_var.getDim(), len(abstract_var.getNodes())), order='F')

                    if markers:
                        ax.plot(range(nodes_var), lb_mat[i, :], marker="x", markersize=3, linestyle='dotted', linewidth=1, color=baseline.get_color())
                        ax.plot(range(nodes_var), ub_mat[i, :], marker="x", markersize=3, linestyle='dotted', linewidth=1, color=baseline.get_color())
                    else:
                        ax.plot(range(nodes_var), lb_mat[i, :], linestyle='dotted')
                        ax.plot(range(nodes_var), ub_mat[i, :], linestyle='dotted')

                    if legend:
                        legend_list.append(f'{abstract_var.getName()}_{i}')
                        legend_list.append(f'{abstract_var.getName()}_{i}_lb')
                        legend_list.append(f'{abstract_var.getName()}_{i}_ub')

        if legend:
            ax.legend(legend_list)

    def plotVariables(self, names=None, grid=False, gather=None, markers=False, show_bounds=True, legend=True, dim=None):

        if self.solution is None:
            raise Exception('Solution not set. Cannot plot variables.')

        if names is None:
            selected_sol = self.solution
        else:
            if isinstance(names, str):
                names = [names]
            selected_sol = {name: self.solution[name] for name in names}

        if gather:

            fig, gs = create_grid(len(selected_sol), 'Variables', gather)
            i = 0
            for key, val in selected_sol.items():
                ax = fig.add_subplot(gs[i, :])
                if grid:
                    ax.grid(axis='x')
                self._plotVar(val, ax, self.prb.getVariables(key), markers=markers, show_bounds=show_bounds, legend=legend, dim=dim)

                # options
                ax.set_title('{}'.format(key))
                ax.ticklabel_format(useOffset=False, style='plain')
                ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
                # ax.set(xlabel='nodes', ylabel='vals')
                # plt.xticks(list(range(val.shape[1])))
                i = i+1
        else:
            for key, val in selected_sol.items():
                fig, ax = plt.subplots(layout='constrained')
                ax.set_title('{}'.format(key))
                if grid:
                    ax.grid(axis='x')
                self._plotVar(val, ax, self.prb.getVariables(key), markers=markers, show_bounds=show_bounds, legend=legend, dim=dim)


        fig.tight_layout()
        plt.show(block=False)

    def plotVariable(self, name, grid=False, markers=None, show_bounds=None, legend=None, dim=None):

        if self.solution is None:
            raise Exception('Solution not set. Cannot plot variable.')

        val = self.solution[name]

        fig, ax = plt.subplots(layout='constrained')
        if grid:
            ax.grid(axis='x')
        self._plotVar(val, ax, self.prb.getVariables(name), markers=markers, show_bounds=show_bounds, legend=legend, dim=dim)

        ax.set_title('{}'.format(name))
        # plt.xticks(list(range(val.shape[1])))
        ax.set(xlabel='nodes', ylabel='vals')

    def plotFunctions(self, grid=False, gather=None, markers=None, show_bounds=None, legend=None, dim=None):

        if self.solution is None:
            raise Exception('Solution not set. Cannot plot functions.')

        if self.prb.getConstraints():
            if gather:
                fig, gs = create_grid(len(self.prb.getConstraints()), 'Functions', gather)

                i = 0
                for name, fun in self.prb.getConstraints().items():
                    ax = fig.add_subplot(gs[i])
                    if grid:
                        ax.grid(axis='x')
                    fun_evaluated = self.prb.evalFun(fun, self.solution)
                    self._plotVar(fun_evaluated, ax, self.prb.getConstraints(name), markers=markers, show_bounds=show_bounds, legend=legend, dim=dim)

                    ax.set_title('{}'.format(name))
                    plt.xticks(list(range(fun_evaluated.shape[1])))
                    ax.ticklabel_format(useOffset=False, style='plain')
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
                    i = i+1

            else:
                for name, fun in self.prb.getConstraints().items():
                    fig, ax = plt.subplots(layout='constrained')
                    ax.set_title('{}'.format(name))
                    if grid:
                        ax.grid(axis='x')
                    fun_evaluated = self.prb.evalFun(fun, self.solution)
                    self._plotVar(fun_evaluated, ax, self.prb.getConstraints(name), markers=markers, show_bounds=show_bounds, legend=legend, dim=dim)

            fig.tight_layout()
            plt.show(block=False)

    def plotFunction(self, name, grid=False, markers=None, show_bounds=None, legend=None, dim=None):

        if self.solution is None:
            raise Exception('Solution not set. Cannot plot functions.')

        fun = self.prb.getConstraints(name)

        fig, ax = plt.subplots(layout='constrained')
        ax.set_title('{}'.format(name))
        if grid:
            ax.grid(axis='x')
        fun_evaluated = self.prb.evalFun(fun, self.solution)
        self._plotVar(fun_evaluated, ax, self.prb.getConstraints(name), markers=markers, show_bounds=show_bounds, legend=legend, dim=dim)

        fig.tight_layout()
        plt.show(block=False)

