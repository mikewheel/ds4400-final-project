"""
Main module for exploratory visualization.

Written by Michael Wheeler and Jay Sherman.
"""
from bokeh.io import export_png
from bokeh.layouts import row
from bokeh.plotting import figure
from contextlib import suppress
from numpy import histogram
from pandas import DataFrame, read_csv
from pandas.core.groupby.generic import DataFrameGroupBy
from statistics import mean, stdev

from config import DATA_DIR

wine_data = {
    "red": read_csv(DATA_DIR / "wine_quality_red.csv"),
    "white": read_csv(DATA_DIR / "wine_quality_white.csv"),
}

wine_features = {  # TODO -- find out the units for these features
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


def generate_feature_histograms_by_category() -> None:
    """
    Generates side-by-side histograms of joint distributions (red vs white) for each feature.
    Leverages example code from https://stackoverflow.com/a/45822244/8857601
    """
    for feature in wine_features:
        plots = {}
        for color, df in wine_data.items():
            binned_data, bin_edges = histogram(df[feature], 20)
            plot_title = f'{feature} ({color}): μ={round(mean(df[feature]), 2)}, σ={round(stdev(df[feature]), 2)}, ' \
                         f'n={len(df[feature])}'
            plot = figure(title=plot_title)
            plot.quad(top=binned_data, bottom=0, left=bin_edges[:-1], right=bin_edges[1:], line_color="white")
            plots[color] = plot
        big_plot = row(*[val for key, val in plots.items()])
        with suppress(KeyboardInterrupt):
            export_png(big_plot, filename=DATA_DIR / f'joint_distribution_{feature}.png')
        

def generate_feature_boxplots_by_quality() -> None:
    """
    Generates box-and-whisker plots for ratings of wine quality across all other features.
    Leverages example code from https://docs.bokeh.org/en/latest/docs/gallery/boxplot.html
    """
    for feature in (wine_features - {"quality"}):
        for color, color_data in wine_data.items():
            
            df = color_data.filter([feature, "quality"], axis="columns")
            quality_groups = df.groupby("quality")
            
            def get_quartiles(groups_: DataFrameGroupBy) -> tuple:
                """
                Determines the 25th, 50th, and 75th percentiles of a feature grouped by wine quality score.
                :param groups_: wine observations grouped by quality score.
                :return: three DataFrames with the quartile boundaries for each group.
                """
                q1 = groups_.quantile(q=0.25)
                q2 = groups_.quantile(q=0.5)
                q3 = groups_.quantile(q=0.75)
                return q1, q2, q3
            
            def get_interquartile_range(groups_: DataFrameGroupBy) -> tuple:
                """
                Determines the interquartile range of a feature grouped by wine quality score.
                :param groups_: wine observations grouped by quality score.
                :return: two dataframes with the lower and upper bounds for each group.
                """
                q1, q2, q3 = get_quartiles(groups_)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                return lower, upper
            
            def get_outliers(groups_: DataFrameGroupBy, feature_: str = feature) -> dict:
                """
                Determines the outliers for the joint distribution of a feature by group.
                :param groups_: A collection of wine observations grouped by quality score.
                :param feature_: The feature to examine for outliers.
                :return: A dict of DataFrames containing the outliers
                """
                lower, upper = get_interquartile_range(groups_)
                
                outliers = {}
                
                for quality_score, data in groups_:
                    outliers[quality_score] = data[(data[feature_] > upper.iloc[quality_score][feature_]) |
                                                   (data[feature_] < lower.iloc[quality_score][feature_])][feature_]
            
                return outliers

            outliers = get_outliers(quality_groups, feature_=feature)

            # Get outlier coordinates
            if not out.empty:
                outx = []
                outy = []
                for keys in out.index:
                    outx.append(keys[0])
                    outy.append(out.loc[keys[0]].loc[keys[1]])
            
            categories = set(df["quality"])
            p = figure(y_range=categories)
            
            # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
            qmin = quality_groups.quantile(q=0.00)
            qmax = quality_groups.quantile(q=1.00)
            upper.score = [min([x, y]) for (x, y) in zip(list(qmax.loc[:, 'score']), upper.score)]
            lower.score = [max([x, y]) for (x, y) in zip(list(qmin.loc[:, 'score']), lower.score)]
            
            # stems
            p.segment(categories, upper.score, categories, q3.score, line_color="black")
            p.segment(categories, lower.score, categories, q1.score, line_color="black")
            
            # boxes
            p.vbar(categories, 0.7, q2.score, q3.score, fill_color="#E08E79", line_color="black")
            p.vbar(categories, 0.7, q1.score, q2.score, fill_color="#3B8686", line_color="black")
            
            # whiskers (almost-0 height rects simpler than segments)
            p.rect(categories, lower.score, 0.2, 0.01, line_color="black")
            p.rect(categories, upper.score, 0.2, 0.01, line_color="black")
            
            # outliers
            if not out.empty:
                p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)
                

if __name__ == "__main__":  # Redundant
    generate_feature_boxplots_by_quality()

