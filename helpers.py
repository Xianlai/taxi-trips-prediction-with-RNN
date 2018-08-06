#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script implements the helper variables and functions.

Author: Xian Lai
Project: NYC taxi pickups pattern learning
Date: Mar. 03, 2018
"""
import os
import gc
import sys
import time
import math
import pickle
import argparse
import numpy as np
import pandas as pd
from os import path
from scipy import stats
from copy import deepcopy
from scipy import sparse
from pprint import pprint
from functools import reduce
from itertools import product
from ipywidgets import interact

from bokeh import palettes
from bokeh.plotting import curdoc, figure
from bokeh.layouts import gridplot, row, column
from bokeh.tile_providers import CARTODBPOSITRON_RETINA
from bokeh.io import output_file, output_notebook, push_notebook, show
from bokeh.models import (
    ColumnDataSource, LinearColorMapper, Quad, FixedTicker, BoxAnnotation, Band
)
# ############################ PROJECT-WIDE VARIABLES ########################
WEEKDAYS = [
    'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'
]
cm_weekdays = {k:v for k, v in zip(WEEKDAYS, palettes.RdYlGn7)}


# ############################ MISC FUNCTIONS ################################

identity_fn     = lambda x: x
flatten         = lambda l: [item for sublist in l for item in sublist]
indices         = lambda seq: list(range(len(seq)))
desparsify_dict = lambda d: {k:v.A for k, v in d.items()}

def take_1st(recording):
    """ take first datapoint out of each batch for every state in recording
    """
    exceptions = ['memory', 'erase', 'write', 'memory_prev', 'encoding_weights', 'ws']
    for k, v in recording.items():
        if (type(v) is np.ndarray) and (k not in exceptions): 
            recording[k] = v[0,]
        if type(v) is list: 
            recording[k] = [x[0,] if type(x) is np.ndarray else x for x in v]
    return recording


def overlay_dataframe(trips, fn):
    """
    """
    hours = trips.copy()
    hours['counts'] = hours['counts'].apply(fn)
    agg = lambda df: {
        'x':list(range(24)), 'y':df['counts'].values, 
        'legend':df['weekday'].iloc[0]
    }
    days = pd.DataFrame(list(hours.groupby('day').apply(agg).values))
    days['color'] = days['legend'].apply(lambda x: cm_weekdays[x])
    
    return days

def prepare_slices(data, z_axis_index):
    """ Plot the first slice of given 3-d array"""
    z_axis_len = data.shape[z_axis_index]
    slices = np.split(data, z_axis_len, z_axis_index)
    slices = [np.squeeze(slice_) for slice_ in slices]

    return slices, z_axis_len

def wrap_index(idx, vector_size):
    """ wrap the index so they always stay inside vector size.
    """
    if idx < 0: return vector_size + idx
    if idx >= vector_size : return idx - vector_size
    else: return idx

def std_scale_list(lst):
    rng_, min_ = max(lst) - min(lst), min(lst)
    return [(x - min_)/rng_ for x in lst]




# ############################ Array Functions #################################
valid_type   = lambda arr: arr.astype(np.float64)
sparsify     = lambda arr: sparse.csr_matrix(np.float64(arr))
minmax_scale = lambda arr: (arr-np.min(arr)) / (np.max(arr)-np.min(arr))

def std_scale_arr(arr):
    std, mean = arr.std(), arr.mean()
    return (arr - mean) / (std + 1e-5)

def minmax_scale_arr(arr):
    """
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def softmax_arr(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid_arr(x):
    """"""
    return 1 / (1 + np.exp(-x))

def normalize_arr(arr):
    return arr / arr.sum()

def take_2d(data, indices=(0, 0), axes=(0, 1)):
    """
    """
    x, y = axes
    data = np.moveaxis(data, y, 0)
    data = np.moveaxis(data, x + 1, 0)
    i, j = indices
    return data[i,j,:,:]




# ############################ PATHS & FILES #################################
proj = "taxi-trips-time-series-prediction"
# proj_path = path.join("/Users/LAI/Documents/data_science/projects/github/", proj)
proj_path = path.join(os.getcwd(), os.pardir)
data_path = path.join(proj_path, "_data/")
log_path  = path.join(proj_path, "_log/")
ckpt_path = path.join(proj_path, "_checkpoint/")
ckpt_file = os.path.join(ckpt_path, "ckpt")

with open(data_path + "grids.pkl", "rb") as f:
    bounds, grids = pickle.load(f)
with open(data_path + "training_ds.pkl", "rb") as f:
    train_features, train_targets = pickle.load(f)
with open(data_path + "testing_ds.pkl", "rb") as f:
    test_features, test_targets = pickle.load(f)

n_hours, n_grids = train_features.shape

def unwrap_outputs(outputs):
    """
    """
    predicts = np.squeeze(np.stack([output[0].A for output in outputs]))
    targets = np.squeeze(np.stack([output[1].A for output in outputs]))
    losses = [output[2] for output in outputs]
    
    return predicts, targets, losses


def unwrap_records(records):
    """
    """
    get_state = lambda k: np.stack([[s[k] for s in seq] for seq in records])
    return {k:get_state(k) for k in records[0][0].keys()}



# ############################ INPUT DATA ####################################
class Dataset():

    """
    The dataset is a 3-d array with axes representing time_bin, grid_y_coor, 
    grid_x_coor.

    values has shape (n_data, hidden_size)
    """

    def __init__(self, features, targets, batch_size=20, sequence_length=24):
        # self.features = minmax_scale_arr(features).astype(np.float64)
        # self.targets  = minmax_scale_arr(targets).astype(np.float64)
        # self.features = softmax_arr(self.features)
        self.features = features.astype(np.float64)
        self.targets  = targets.astype(np.float64)
        self.targets_rng = 5632.0

        self.n_data, self.input_size = features.shape
        self.batch_size = batch_size
        self.seq_len    = sequence_length
        self.output_size = 1

    def next_batch(self, batch_size=None, random_seed=None):
        """
        """
        if batch_size == None: batch_size = self.batch_size

        # the slicing start positions of this batch
        if random_seed is not None: np.random.seed(seed=random_seed)
        starts = np.random.randint(
            0, self.n_data - self.seq_len - 1, size=batch_size
        )
        ends   = [start + self.seq_len for start in starts]
        ranges = list(zip(starts, ends))
        # slice the input sequences with sequence length and stack them as 1 
        # array with shape: (batch_sz, sequence_length, input_shape)
        # re-order the axis as (sequence_length, batch_sz, input_shape)
        inputs = np.stack([self.features[r[0]:r[1]] for r in ranges])
        inputs = np.moveaxis(inputs, source=1, destination=0)

        return inputs, ranges

    def get_target(self, ranges):
        """
        """
        locs = [r[1] + 1 for r in ranges]
        target_batch = [self.targets[loc] for loc in locs]

        return np.stack(target_batch).reshape(-1, 1)


# ############################ TIME MANAGEMENT ###############################
timer  = lambda: time.time()
second = lambda x: str(round(x,2)) + "sec"
minute = lambda x: str(int(x//60)) + "min "
hour   = lambda x: str(int(x//(60*60))) + "hr "

def elapsed(sec):
    if sec<60: return second(sec)
    elif sec<(60*60): return minute(sec) + second(sec % 60)
    else: return hour(sec) + minute(sec % (60*60))


# ############################ PLOTTING ######################################
grey = {
    'white':'#ffffff', 'light':'#efefef', 'median':'#aaaaaa', 
    'dark':'#282828', 'black':'#000000'
}

plasma = palettes.Plasma256
greys  = palettes.Greys256

# ---------------------------- FIGURE SETTING FUNCTIONS ----------------------
def set_tickers(p, x_range=None, y_range=None, n_x_tickers=None, 
                n_y_tickers=None):
    """ Set the number of tickers"""
    def make_range(rng, n_step):
        return list(range(rng[0], rng[1], (rng[1] - rng[0])//n_step))

    if n_x_tickers: p.xaxis.ticker = make_range(x_range, n_x_tickers)
    if n_y_tickers: p.yaxis.ticker = make_range(y_range, n_y_tickers)

    return p

def set_axes_vis(p, xlabel, ylabel):
    """ Set the visibility of axes"""
    if not xlabel: p.xaxis.visible = False  
    if not ylabel: p.yaxis.visible = False

    return p

def hide_toolbar(p):
    """ Set whether show toolbar"""
    p.toolbar.logo = None
    p.toolbar_location = None

    return p

def set_legend(p, location="top_right", click_policy=None, line_alpha=1, 
                fill_alpha=1):
    """ """
    p.legend.location = location
    p.legend.click_policy = click_policy
    p.legend.border_line_alpha = line_alpha
    p.legend.background_fill_alpha = fill_alpha

    return p

def hide_grid(p):
    """ """
    p.xgrid.visible = False
    p.ygrid.visible = False

    return p


# ---------------------------- IMAGE PLOTTING --------------------------------
def plot_image(data, f_w=900, f_h=600, xlabel=None, ylabel=None, title=None, 
               n_x_tickers=None, n_y_tickers=None, color_range=(0, 1),  
               transpose=True, use_toolbar=True, silent=False, palette=plasma):
    """ plot the image using given data.
    """
    if len(data.shape) == 1: data = data.reshape(1, -1)
    if transpose: data = data.T
    h, w = data.shape
    
    p = figure(
        title=title, 
        plot_width=f_w, plot_height=f_h, 
        x_range=(0, w), y_range=(0, h),
        x_axis_label=xlabel,
        y_axis_label=ylabel
    )
    if not use_toolbar: p = hide_toolbar(p)
    p = set_axes_vis(p, xlabel, ylabel)
    # p = set_tickers(p, (0, w), (0, h), n_x_tickers, n_y_tickers)

    cds = ColumnDataSource(data={'image':[data]})
    cm = LinearColorMapper(
        palette=palette, low=color_range[0], high=color_range[1]
    )
    r = p.image(
        image='image', 
        x=0, y=0, dw=w, dh=h,
        color_mapper=cm,
        source=cds
    )
    
    if silent: return p, r
    else: handle = show(p, notebook_handle=True)
        
    def update(rescale=False):
        if rescale: 
            data_rescaled = minmax_scale(data)
            r.data_source.data = {'image':[data_rescaled]}
            push_notebook(handle=handle)
        else:
            r.data_source.data = {'image':[data]}
            push_notebook(handle=handle)

    interact(update, rescale=False)
   

def plot_video(data, time_axis, f_w=900, f_h=600, transpose=True, 
               xlabel=None, ylabel=None, title=None, sleep_time=0.2,
               n_x_tickers=None, n_y_tickers=None, palette=plasma):
    """ plot a video using given 3d array.

    Args:
        data (3d array): the data source of video
        time_axis (int): the axis of time
    """
    slices, z_axis_len = prepare_slices(data, time_axis)
    p, r = plot_image(
        slices[0], f_w, f_h, xlabel, ylabel, title=title + ':' + str(0), 
        n_x_tickers=n_x_tickers, n_y_tickers=n_y_tickers, silent=True, 
        transpose=transpose, palette=palette
    )
    handle = show(p, notebook_handle=True)
    for i in range(1, z_axis_len):
        img = slices[i].T if transpose else slices[i]
        if len(img.shape) == 1: img = img.reshape(-1, 1)
        r.data_source.data = {'image':[img]}
        p.title.text = title + ':' + str(i)
        push_notebook(handle=handle)
        time.sleep(sleep_time)


def plot_slides(data, slide_axis, f_w=900, f_h=600, xlabel=None, ylabel=None, 
                title=None, transpose=True, color_range=(0, 1), n_x_ticks=None, 
                n_y_ticks=None, palette=plasma):
    """ plot slides controlled by interactive slider.
    """
    slices, slide_axis_len = prepare_slices(data, slide_axis)
    p, r = plot_image(
        slices[0], f_w, f_h, title=title + ':' + str(0), xlabel=xlabel, 
        ylabel=ylabel, silent=True, transpose=transpose, palette=palette,
        n_x_tickers=n_x_ticks, n_y_tickers=n_y_ticks, color_range=color_range,
    )
    handle = show(p, notebook_handle=True)
    
    def update(i=0, rescale=False):
        img = slices[i].T if transpose else slices[i]
        if len(img.shape) == 1: img = img.reshape(-1, 1)
        r.data_source.data = {'image':[img]}
        p.title.text = title + ':' + str(i)

        if rescale: 
            img_rescaled = minmax_scale(img)
            r.data_source.data = {'image':[img_rescaled]}
        else:
            r.data_source.data = {'image':[img]}

        push_notebook(handle=handle)

    interact(update, i=(0, slide_axis_len - 1, 1), rescale=False)


def plot_slides_4d(data, slide_axes, f_w=900, f_h=600, xlabel=None, ylabel=None, 
                   title=None, transpose=True, n_x_ticks=None, n_y_ticks=None,
                   color_range=(0, 1), palette=plasma):
    """ plot slides controlled by 2 interactive sliders representing 2 axes.
    """
    slice_ = lambda indices: take_2d(data, indices, slide_axes)

    p, r = plot_image(
        slice_((0, 0)), f_w, f_h, title=title + ':' + str([0, 0]), xlabel=xlabel, 
        ylabel=ylabel, silent=True, transpose=transpose, color_range=color_range,
        n_x_tickers=n_x_ticks, n_y_tickers=n_y_ticks, palette=palette,
    )
    handle = show(p, notebook_handle=True)
    
    def update(i=0, j=0, rescale=False):
        img = slice_((i, j)).T if transpose else slice_((i, j))
        if len(img.shape) == 1: img = img.reshape(-1, 1)
        r.data_source.data = {'image':[img]}
        p.title.text = title + ':' + str([i, j])

        if rescale: 
            img_rescaled = minmax_scale(img)
            r.data_source.data = {'image':[img_rescaled]}
        else:
            r.data_source.data = {'image':[img]}

        push_notebook(handle=handle)

    imax, jmax = data.shape[slide_axes[0]], data.shape[slide_axes[1]]
    interact(update, i=(0, imax - 1, 1), j=(0, jmax - 1, 1), rescale=False)


def plot_records_dynamic(f_states, c_states, h_states, o_states, targets, 
                         f_w=250, sleep_time=0.5, color_range=(0, 1)):
    """
    """
    pm_0 = {'f_w':f_w, 'f_h':600, 'silent':True, 'use_toolbar':False, 
            'xlabel':"sequence_step", 'color_range':color_range}
    pm_1 = {'silent':True, 'f_h':600, 'use_toolbar':False, 
            'color_range':color_range}
    ylabel, ite = "hidden_layer", "_state: iteration: -1"

    p_o, r_o = plot_image(o_states[0], f_w=90, ylabel=ylabel, **pm_1)
    p_t, r_t = plot_image(targets[0], f_w=43, **pm_1)

    p_f, r_f = plot_image(f_states[0], title="f"+ite, **pm_0)
    p_c, r_c = plot_image(c_states[0], title="c"+ite, **pm_0)
    p_h, r_h = plot_image(h_states[0], title="h"+ite, **pm_0)
    
    handle   = show(row([p_o, p_t, p_f, p_c, p_h]), notebook_handle=True)
    
    for i in range(1, targets.shape[0]):
        r_f.data_source.data = {'image':[f_states[i].T]}
        r_c.data_source.data = {'image':[c_states[i].T]}
        r_h.data_source.data = {'image':[h_states[i].T]}
        r_o.data_source.data = {'image':[o_states[i].reshape(-1, 1)]}
        r_t.data_source.data = {'image':[targets[i].reshape(-1, 1)]}
        p_f.title.text = "j_input: iteration: %d" %i
        p_c.title.text = "long_term: iteration: %d" %i
        p_h.title.text = "short_term: iteration: %d" %i
        push_notebook(handle=handle)
        time.sleep(sleep_time)


# ---------------------------- GRIDS PLOTTING --------------------------------
def grids_data_source(values, grids):
    """ Prepare the data source for grid plotting.
    Args:
        values: an array with the same shape as grids
        grids: the bounds for each grid
    Returns: 
        the data source for grid plotting as a dictionary 
    """
    # grid boundaries
    left, bottom, right, top = (list(bound) for bound in zip(*grids))
    
    # grid colors and alphas
    values = minmax_scale(values)
    _, bins = np.histogram(values, bins=80)
    indices = np.digitize(values, bins)
    alphas = [value*3 for value in values]
    
    return {
        'left':left, 'bottom':bottom, 'right':right, 'top':top, 
        'color':indices, 'alpha':alphas
    }

def plot_grids(values, grids, title=None, f_w=1000, f_h=1000, silent=False):
    """ plot the grids on map tile background.

    Args:
        values (array): either the input data or output data as a 1-d array
        grids (list): the bounds of each grid

    """
    values = values
    cds    = ColumnDataSource(data=grids_data_source(values, grids))
    cm     = LinearColorMapper(palette=palettes.grey(10))
    center = (-8231000.0 - 3000, 4977500.0 - 2000)
    view_w = 7000
    view_h = 5000
    p = figure(
        title=title,
        plot_width=f_w, 
        plot_height=f_h,
        x_range=(center[0] - view_w,center[0] + view_w), 
        y_range=(center[1] - view_h,center[1] + view_h),
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    p.add_tile(CARTODBPOSITRON_RETINA)
    p.axis.visible = False

    r = p.quad(
        left='left', right='right', top='top', bottom='bottom', source=cds,
        line_alpha=0, alpha='alpha', color={'field': 'color', 'transform': cm}
    )

    if silent: return p, r
    else: show(p)

def plot_grids_dynamic(values_seq, grids, f_w=300, f_h=200, sleep_time=0.2):
    """ Plot the grids of multiple time steps.
    """
    values_seq = values_seq
    p, r   = plot_grids(values_seq[0], grids, f_w=f_w, f_h=f_h, silent=True)
    handle = show(p, notebook_handle=True)

    for values in values_seq[1:]:
        r.data_source.data = grids_data_source(values, grids)
        push_notebook(handle=handle)
        time.sleep(sleep_time)


def compare_grids_slide(values_seq_0, values_seq_1, grids, f_w=300, f_h=200,
                        title_0="predictions", title_1="targets"):
    """ plot 2 grids side by side"""
    p0, r0 = plot_grids(
        values_seq_0[0], grids, title=title_0, f_w=f_w, f_h=f_h, silent=True
    )
    p1, r1 = plot_grids(
        values_seq_1[0], grids, title=title_1, f_w=f_w, f_h=f_h, silent=True
    )
    p0 = hide_toolbar(p0); p1 = hide_toolbar(p1)
    handle = show(column(p0, p1), notebook_handle=True)

    def update(i=0):
        r0.data_source.data = grids_data_source(values_seq_0[i], grids)
        r1.data_source.data = grids_data_source(values_seq_1[i], grids)
        push_notebook(handle=handle)

    interact(update, i=(0, len(values_seq_0) - 1, 1))


# ---------------------------- LINES PLOTTING --------------------------------
def plot_line(x, y, x_range=None, y_range=None, title=None, f_w=300, f_h=300, 
              silent=False, use_toolbar=True):
    """
    """
    p = figure(
        title=title, x_range=x_range, y_range = y_range, 
        plot_width=f_w, plot_height=f_h
    )
    p.background_fill_color = 'white'
    p = hide_grid(p)
    r = (p.line(x=x, y=y, color='black'), p.circle(x=x, y=y, color='black'))
    if not use_toolbar: p = hide_toolbar(p)

    if silent: return p, r
    else: show(p)

def plot_lines(xs, ys, legends, x_range=None, y_range=None, title=None,   
               f_w=600, f_h=300, silent=False, bg='black'):
    """ just plot n lines with different color.
    """
    p = figure(
        title=title, x_range=x_range, y_range=y_range, 
        plot_width=f_w, plot_height=f_h
    )
    p.background_fill_color = bg
    colors = palettes.viridis(len(xs))
    for x, y, lg, color in zip(xs, ys, legends, colors):
        r = p.line(x=x, y=y, legend=lg, color=color)
    if silent: return p, r
    else: show(p)
    
def overlay_durations(df, x_range=None, y_range=None, f_w=600, f_h=300, 
                      title=None, silent=False):
    """
    """
    p = figure(
        title=title, x_range=x_range, y_range=y_range, 
        plot_width=f_w, plot_height=f_h
    )
    grps = df.groupby('legend')
    for legend, grp in grps:
        r = p.multi_line(
            xs=grp['x'], ys=grp['y'], color=grp['color'], 
            line_width=0.5, legend=legend
        )
    p.background_fill_color = 'black'
    p = set_legend(
        p, location="top_left", click_policy="hide", line_alpha=0, 
        fill_alpha=0
    )
    p = hide_grid(p)

    if silent: return p, r
    else: show(p)


def plot_states(states, legends, title=None, f_w=900, f_h=500, y_range=None):
    """ plot given multiple states with different color
    """
    i, s, u = states[legends[0]].shape
    p = figure(
        title=title, x_range=(0, s), x_axis_label="seq_step", y_range=y_range,
        y_axis_label="intensity", plot_width=f_w, plot_height=f_h
    )
    p.background_fill_color = 'black'
    p = hide_grid(p)
    p = set_tickers(p, x_range=(0, s), n_x_tickers=s)

    rs= []
    n_states = len(legends)
    if n_states < 3: colors = palettes.Spectral[3][:n_states]
    else: colors = palettes.Spectral[n_states]
    for legend, color in zip(legends, colors):
        state = states[legend]
        if legend == 'ea_states':
            cds = ColumnDataSource(data={
                'x':np.arange(0, s, 0.5) - 0.5, 
                'y':state[0,:,0].copy()
            })
        else:
            cds = ColumnDataSource(data={
                'x':np.arange(0, s, 1), 'y':state[0,:,0]
            })
        rs.append((
            p.circle(x='x', y='y', source=cds, legend=legend, color=color), 
            p.line(x='x', y='y', source=cds, legend=legend, color=color)
        ))
        
    handle = show(p, notebook_handle=True)
    p = set_legend(
        p, location="top_left", click_policy="hide", line_alpha=0, 
        fill_alpha=0
    )
    def update(iteration=0, state_unit=0):
        for r, l in zip(rs, legends): 
            r[0].data_source.data['y'] = states[l][iteration,:,state_unit]
            r[1].data_source.data['y'] = states[l][iteration,:,state_unit]
        push_notebook(handle=handle)

    interact(update, iteration=(0, i - 1, 1), state_unit=(0, u - 1, 1))



def overlay_durations_slide(trips, x_range=None, y_range=None, title=None, 
                            f_w=950, f_h=600):
    """
    """
    p = figure(
        title=title, x_range=x_range, y_range=y_range, 
        plot_width=f_w, plot_height=f_h
    )

    rs = []
    grps = overlay_dataframe(trips, lambda x: x[3315]).iloc[:200]\
        .groupby('legend')
    for legend, grp in grps:
        rs.append(p.multi_line(
            xs=grp['x'], ys=grp['y'], color=grp['color'], 
            line_width=0.5, legend=legend
        ))
    p.background_fill_color = 'black'
    p = set_legend(
        p, location="top_left", click_policy="hide", line_alpha=0, 
        fill_alpha=0
    )
    p = hide_grid(p)
    handle = show(p, notebook_handle=True)
    
    def update(grid_idx=2778):
        grps = overlay_dataframe(trips, lambda x: x[grid_idx]).iloc[:200]\
            .groupby('legend')
        cnt = 0
        for _, grp in grps:
            rs[cnt].data_source.data['ys'] = grp['y']
            cnt += 1
        push_notebook(handle=handle)

    interact(update, grid_idx=(0, n_grids - 1, 1))


def process_perf_files():
    def fetch_losses(path):
        f = open(path, "rb")
        _, _, losses = unwrap_outputs(pickle.load(f)[1])

        return losses
    
    perfs_path = data_path + "perfs/"
    perfs_files = [os.path.join(perfs_path, f) for f in os.listdir(perfs_path)]
    legends = [name[70:-4] for name in perfs_files]
    losses = [fetch_losses(f) for f in perfs_files]
    
    return losses, legends


def compare_perfs_cells(ys, legends, x_range=(0, 150), y_range=None, 
                         f_w=900, f_h=900, title=None, silent=False):
    """
    """
    p = figure(
        title=title, x_range=x_range, y_range=y_range, 
        plot_width=f_w, plot_height=f_h
    )
    xs = [range(150)] * len(ys)
    colors = palettes.Greens[9] + palettes.Blues[7] + palettes.Oranges[9]
    
    for x, y, legend, color in zip(xs, ys, legends, colors):
        p.circle(x=x, y=y, legend=legend, color=color)
        p.line(x=x, y=y, legend=legend, color=color)

    p.background_fill_color = 'black'
    p = hide_grid(p)
    p = set_legend(
        p, location="top_right", click_policy="hide", line_alpha=0, 
        fill_alpha=0
    )
    show(p)


def compare_perfs_layers(ys, legends, x_range=(0, 150), y_range=None, f_w=900,
                         f_h=900, title=None, palette=palettes.viridis,
                         silent=False):
    """
    """
    p = figure(
        title=title, x_range=x_range, y_range=y_range, 
        plot_width=f_w, plot_height=f_h
    )
    n = len(ys)
    
    for x, y, legend, color in zip([range(150)] * n, ys, legends, palette(n)):
        p.circle(x=x, y=y, legend=legend, color=color)
        p.line(x=x, y=y, legend=legend, color=color)

    p.background_fill_color = 'black'
    p = hide_grid(p)
    p = set_legend(
        p, location="top_right", click_policy="hide", line_alpha=0, 
        fill_alpha=0
    )
    show(p)
