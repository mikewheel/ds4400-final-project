"""
Main module for exploratory visualization.

Written by Michael Wheeler and Jay Sherman.
"""
from contextlib import suppress
from statistics import mean, stdev

from bokeh.io import export_png
from bokeh.layouts import row
from bokeh.plotting import figure
from numpy import histogram
from pandas import read_csv

from config import INPUT_DATA_DIR, VISUALIZATION_DATA_DIR

wine_data = {
    "red": read_csv(INPUT_DATA_DIR / "wine_quality_red.csv"),
    "white": read_csv(INPUT_DATA_DIR / "wine_quality_white.csv"),
}

wine_features = {
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality"
}

for feature in wine_features:
    plots = {}
    for color, df in wine_data.items():
        # Shoutout to https://stackoverflow.com/a/45822244/8857601
        binned_data, bin_edges = histogram(df[feature], 20)
        plot_title = f'{feature} ({color}): μ={round(mean(df[feature]), 2)}, σ={round(stdev(df[feature]), 2)}, ' \
                     f'n={len(df[feature])}'
        plot = figure(title=plot_title)
        plot.quad(top=binned_data, bottom=0, left=bin_edges[:-1], right=bin_edges[1:], line_color="white")
        plots[color] = plot
    big_plot = row(*[val for key, val in plots.items()])
    with suppress(KeyboardInterrupt):
        export_png(big_plot, filename=VISUALIZATION_DATA_DIR / f'joint_distribution_{feature}.png')
        pass
