'''
Generate a graph of geodata point count per partition region
to the total number of some measurement in the region.
'''

import logging
import argparse
import numpy as np
import pandas as pd
import scipy.spatial
import scipy.optimize
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from sympy import expand
from sympy.abc import x as x_symbol
from shapely.geometry import Point
from utils import init_logger, partition, get_geo_stepsizes

logger = init_logger()
parser = argparse.ArgumentParser(description='Generate a graph of geodata point count per partition region to the total number of some measurement in the region.')
parser.add_argument('input', type=str, help='The path to the input geodata.')
parser.add_argument('y_columns', type=str, nargs='+', help='The (case-sensitive) header names of the variable columns.')
parser.add_argument('--longitude-col', dest='longitude_column', type=str, help='The (case-sensitive) header name of the longitude column.', default='LONGITUDE')
parser.add_argument('--latitude-col', dest='latitude_column', type=str, help='The (case-sensitive) header name of the latitude column.', default='LATITUDE')
parser.add_argument('--count-adjustment-col', dest='count_adjustment_column', type=str, help='The (case-sensitive) header name of the count adjustement column')
parser.add_argument('-cw', '--chunk-width', type=int, help='The number of chunks on the longitude (horizontal). Defaults to 32.', default=32)
parser.add_argument('-ch', '--chunk-height', type=int, help='The number of chunks on the latitude (vertical). Defaults to 32.', default=32)
parser.add_argument('--no-plot-lin-reg', dest='plot_linreg', action='store_false', help='Disables a linear regression.')
parser.add_argument('--plot-log-reg', dest='plot_logreg', action='store_true', help='Enables a logarithmic regression.')
parser.add_argument('--matplotlib-style', type=str, help='The matplotlib graph style.', default='default')
parser.add_argument('--title', type=str, help='The title of the graph.')
parser.add_argument('--x-label', type=str, help='The label of the x-axis.', default='X')
parser.add_argument('--y-label', type=str, help='The label of the y-axis', default='Y')
parser.add_argument('--scatter-label', type=str, help='The scatter plot label.', default='')
parser.add_argument('--plot-label', type=str, help='The plot label.', default='')
parser.add_argument('--export', dest='export', help='Enable export to file.', action='store_true')
parser.add_argument('--no-export', dest='export', help='Disable export to file.', action='store_false')
parser.add_argument('--export-output', type=str, help='The path to the exported file.', default=None)
parser.add_argument('--export-dpi', type=int, help='The DPI of the exported file.', default=400)
parser.add_argument('--export-format', type=str, help='The format of the exported file.', default='png')
parser.add_argument('--no-preview', dest='preview', help='Disable the graph preview window.', action='store_false')
parser.add_argument('--export-csv', dest='export_csv', action='store_true', help='Output the data to a CSV.')
parser.add_argument('--csv-directory', help='The CSV output directory.', default='.')
parser.set_defaults(export=False, preview=True, plot_linreg=True, plot_logreg=False, export_csv=False)
args = parser.parse_args()

input_path = Path(args.input)
if not (input_path.is_file() or input_path.exists()):
    logger.error('The specified input is not a file or does not exist!')
    exit(1)

def coefficient_of_determination(X, y, trendline, weights=None):
    '''
    Calculate the coefficient of determination (R-squared value).
    '''

    if weights is None:
        # All points are equally weighted
        weights = [1] * len(X)

    y_regression = trendline(X)  
    mean = np.average(y, weights=weights)

    # Calculate the total sum of squares (tss) and
    # the residual sum of squares.
    tss = np.sum(weights * (y - mean)**2)
    rss = np.sum(weights * (y_regression - y)**2)

    return 1 - rss / tss

def covariance(X, y, weights=None):
    '''
    Calculate the covariance.
    '''

    if weights is None:
        weights = [1] * len(X)
    
    return np.average((X - np.average(X, weights=weights)) * (y - np.average(y, weights=weights)), weights=weights)

def correlation_coefficient(X, y, weights=None):
    '''
    Calculates the Pearson correlation coefficient.
    '''

    if weights is None:
        weights = [1] * len(X)

    return covariance(X, y, weights) / np.sqrt(covariance(X, X, weights) * covariance(y, y, weights))

def _generate_trendline_metrics(X, y, trendline, weights=None):
    p_value = correlation_coefficient(X, y, weights)
    r_squared = coefficient_of_determination(X, y, trendline, weights)
    return p_value, r_squared

def generate_polynomial_trendline(X, y, weights=None, degree=1):
    '''
    Generate a polynomial fit trendline.
    '''

    trendline = np.poly1d(np.polyfit(X, y, degree, w=weights))
    p_value, r_squared = _generate_trendline_metrics(X, y, trendline, weights)
    return trendline, p_value, r_squared

def generate_curve_fit(X, y, f, weights=None, **kwargs):
    '''
    Generate a trendline for any function.
    '''

    popt, pcov = scipy.optimize.curve_fit(f, X, y, sigma=weights, absolute_sigma=weights is not None, **kwargs)
    trendline = lambda x: f(x, *popt)
    p_value, r_squared = _generate_trendline_metrics(X, y, trendline, weights)
    return trendline, p_value, r_squared

df = pd.read_csv(input_path)
points_geometry = [Point(xy) for xy in zip(df[args.longitude_column], df[args.latitude_column])]
chunks = partition(points_geometry, args.chunk_width, args.chunk_height)

X, Y = [], {y_column: list() for y_column in args.y_columns}
for i in range(args.chunk_width):
    for j in range(args.chunk_height):
        x = len(chunks[i][j])
        if x == 0: continue
        
        total_v = {y_column: 0 for y_column in args.y_columns}
        for point in chunks[i][j]:
            if args.count_adjustment_column is not None:
                adjustement_columns = df[(df[args.longitude_column] == point.x) & (df[args.latitude_column] == point.y)][args.count_adjustment_column]
                x += sum(adjustement_columns)

            columns = df[(df[args.longitude_column] == point.x) & (df[args.latitude_column] == point.y)][args.y_columns]
            for column in columns:
                total_v[column] += sum(v if not np.isnan(v) else 0 for v in columns[column])
                
        X.append(x)
        for column in total_v:
            Y[column].append(total_v[column])

X = np.array(X)

# Sort the Xs so that matplotlib can properly display them.
sorted_X = np.array(sorted(X))
logger.setLevel(logging.INFO)

csv_directory = Path(args.csv_directory)
if csv_directory.is_file():
    csv_directory = csv_directory.parent()

if args.export_csv:
    csv_directory.mkdir(parents=True, exist_ok=True)

longitude_stepsize, latitude_stepsize = get_geo_stepsizes(points_geometry, args.chunk_width, args.chunk_height)
chunk_area = longitude_stepsize * latitude_stepsize
chunk_perimeter = 2 * (longitude_stepsize + latitude_stepsize)

X_density = []
for i in range(args.chunk_width):
    for j in range(args.chunk_height):
        points = [(p.x, p.y) for p in chunks[i][j]]
        if len(points) == 0: continue

        unique_length = len(set(points))
        if unique_length == 1:
            density = 0
        elif unique_length == 2:
            p1, p2 = points[0], points[1]
            density = 1 - ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5 / chunk_perimeter
        else:
            hull = scipy.spatial.ConvexHull(points)
            density = 1 - hull.volume / chunk_area
        
        X_density.append(density)

X_density = np.array(X_density)
for column in Y:
    y = Y[column] = np.array(Y[column])
    if args.export_csv:
        csv_output_directory = csv_directory / '{}_{}_processed.csv'.format(input_path.stem, column)
        pd.DataFrame({args.x_label: X, args.y_label: y}).to_csv(csv_output_directory, index=False)

    plt.scatter(X, y, label=args.scatter_label)

    plot_label_template = '{} {{}}'.format(args.plot_label).strip()
    if args.plot_linreg:
        unweighted_trendline, unweighted_p, unweighted_rsquared = generate_polynomial_trendline(X, y)
        plt.plot(sorted_X, unweighted_trendline(sorted_X), linestyle='dashed', label=plot_label_template.format('{} Unweighted (linear)'.format(column)))
        logger.info(plot_label_template.format('{} - Unweighted R-squared (linear): {}'.format(column, round(unweighted_rsquared, 3))))

        unweighted_trendline_str = str(expand(unweighted_trendline(x_symbol)))
        logger.info(plot_label_template.format('{} - Unweighted Trendline (linear): y = {}'.format(column, unweighted_trendline_str)))

        weighted_trendline, weighted_p, weighted_rsquared = generate_polynomial_trendline(X, y, 1 / X_density**2)
        plt.plot(sorted_X, weighted_trendline(sorted_X), linestyle='dashed', label=plot_label_template.format('{} Weighted (linear)'.format(column)))
        logger.info(plot_label_template.format('{} - Weighted R-squared (linear): {}'.format(column, round(weighted_rsquared, 3))))
      
        weighted_trendline_str = str(expand(weighted_trendline(x_symbol)))
        logger.info(plot_label_template.format('{} - Weighted Trendline (linear): y = {}'.format(column, weighted_trendline_str)))

        logger.info(plot_label_template.format('{} - Unweighted Correlation coefficient (linear): {}'.format(column, round(unweighted_p, 3))))    
        logger.info(plot_label_template.format('{} - Weighted Correlation coefficient (linear): {}'.format(column, round(weighted_p, 3))))   

    if args.plot_logreg:
        log_trendline, log_p, log_rsquared = generate_curve_fit(X, y, lambda t, a, b, c: a * np.log(b * t) + c)
        plt.plot(sorted_X, log_trendline(sorted_X), linestyle='dashed', label=plot_label_template.format('{} (logarithmic)'.format(column)))
        logger.info(plot_label_template.format('{} - R-squared (logarithmic): {}'.format(column, round(log_rsquared, 3)))) 

matplotlib.style.use(args.matplotlib_style)
plt.title(args.title)
plt.xlabel(args.x_label)
plt.ylabel(args.y_label)
plt.legend(loc='upper right')

if args.preview:
    if not args.export:
        plt.show()
    else:
        logger.warning('Graph preview was enabled but it could not be displayed since export was also enabled.' +
            ' Previewing and exporting cannot both be enabled.')

if args.export:
    output_format = args.export_format[1:] if args.export_format.startswith('.') else args.export_format
    if args.export_output is None:
        output_extension = output_format
        if output_extension == 'latex':
            output_extension = 'tex'
        
        output_path = input_path.with_suffix('.export.' + output_extension)
    else:
        output_path = Path(args.export_output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == 'latex':
        import tikzplotlib
        tikzplotlib.save(output_path)
    else:
        plt.savefig(output_path, dpi=args.export_dpi)
