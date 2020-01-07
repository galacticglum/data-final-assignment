'''
Generate a map given a set of shapefiles.
'''

import json
import glob
import argparse
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Point
from utils import init_logger, partition

logger = init_logger()
parser = argparse.ArgumentParser(description='Generate a map given a set of shapefiles.')
parser.add_argument('inputs', type=str, nargs='+', help='The input shapefiles or layer descriptor files (JSON). ' +
                                                        'Note: a layer descriptor file must have the JSON file extension.')
parser.add_argument('--points', dest='points_filepath', type=str, help='The path to the CSV file containing the geodata of points on a map.')
parser.add_argument('--points-label', type=str, help='The label of the geodata points.', default=None)
parser.add_argument('--longitude-col', dest='longitude_column', type=str, help='The (case-sensitive) header name of the longitude column.', default='LONGITUDE')
parser.add_argument('--latitude-col', dest='latitude_column', type=str, help='The (case-sensitive) header name of the latitude column.', default='LATITUDE')
parser.add_argument('--count-adjustment-col', dest='count_adjustment_column', type=str, help='The (case-sensitive) header name of the count adjustement column')
parser.add_argument('--crs', type=str, help='The coordinate reference system of the geodata.', default='WGS84')
parser.add_argument('--disable-zordering', dest='use_zordering', help='Order the layers in ascending order (order: map layers and then point layer).', action='store_false')
parser.add_argument('--title', type=str, help='The title of the graph.')
parser.add_argument('--title-padding', type=float, help='The padding on the title of the graph.', default=0)
parser.add_argument('--colour', type=str, help='The colour of the plot.')
parser.add_argument('--xlabel', type=str, help='The label on the x-axis.', default='Latitude')
parser.add_argument('--ylabel', type=str, help='The label on the y-axis.', default='Longitude')
parser.add_argument('--partition', dest='partition_map', help='Partition the data points on the map into chunks.', action='store_true')
parser.add_argument('-cw', '--chunk-width', type=int, help='The number of chunks on the longitude (horizontal). Defaults to 32.', default=32)
parser.add_argument('-ch', '--chunk-height', type=int, help='The number of chunks on the latitude (vertical). Defaults to 32.', default=32)
parser.add_argument('--export', dest='export', help='Enable export to file.', action='store_true')
parser.add_argument('--no-export', dest='export', help='Disable export to file.', action='store_false')
parser.add_argument('--export-output', type=str, help='The path to the exported file.', default=None)
parser.add_argument('--export-dpi', type=int, help='The DPI of the exported file.', default=400)
parser.add_argument('--export-format', type=str, help='The format of the exported file.', default='png')
parser.add_argument('--no-preview', dest='preview', help='Disable the graph preview window.', action='store_false')
parser.add_argument('--show-text', dest='show_text', help='Enable the count text.', action='store_true')
parser.add_argument('--point-size', type=float, help='The scatter plot point size.', default=20)
parser.set_defaults(partition=False, use_zordering=True, export=False, preview=True, show_text=False)
args = parser.parse_args()

matplotlib.rc('text', usetex=True)

if len(args.inputs) == 0:
    logger.error('No shapefile inputs were specified.')
    exit(1)

class Layer:
    def __init__(self, shapefile, linewidth=None, colour=None, **kwargs):
        '''
        Initialize the map layer.
        '''

        self.shapefile = shapefile
        self.linewidth = linewidth
        self.colour = colour or args.colour

    def plot(self, **kwargs):
        '''
        Plot the map layer.
        '''

        self.shapefile.plot(linewidth=self.linewidth, color=self.colour, **kwargs)

def verify_shapefile_path(filepath):
    '''
    Verifies that the specified path to the shapefile is valid.

    :param filepath:
        The path to the shapefile.
    
    :returns:
        A boolean indicating whether the path is valid.

    '''

    filepath = Path(filepath)
    if not (filepath.exists() or filepath.is_file()):
        logger.warn('The specified shapefile input (\'{}\') is not a file or does not exist! Skipping this shapefile.'.format(str(filepath)))
        return False

    return True

def load_layer_description(filepath):
    '''
    Loads the layer description JSON file.

    :param filepath:
        The path to the layer description file.

    :returns:
        A list of Layer objects.
    
    '''

    result = []
    filepath = Path(filepath).resolve().absolute()
    with open(filepath, 'r') as file:
        layer_description = json.load(file)
        for layer in layer_description:
            shapefile_path = filepath.parent / Path(layer['shapefile'])

            if not verify_shapefile_path(shapefile_path): continue
            shapefile = gpd.read_file(shapefile_path)
            linewidth = layer.get('linewidth', None)
            colour = layer.get('colour', None)

            result.append(Layer(shapefile, linewidth, colour))

    return result

# Load the shapefiles...
layers = []
for input_pattern in args.inputs:
    files = glob.glob(input_pattern, recursive=True)
    for input_path in files:
        filepath = Path(input_path)
        # Check if the provided file is a layer description.
        # A layer description file MUST have the JSON extension.
        if filepath.suffix == '.json':
            layers.extend(load_layer_description(filepath))
        else:
            if not verify_shapefile_path(filepath): continue
            shapefile = gpd.read_file(filepath)
            layers.append(Layer(shapefile))

# Plot all the layers
fig, ax = plt.subplots()
for i in range(len(layers)):
    z_index = i if args.use_zordering else None
    layers[i].plot(ax=ax, zorder=z_index)

# Plot and partition points
if args.points_filepath is not None:
    points_filepath = Path(args.points_filepath).resolve().absolute()
    if points_filepath.exists() and points_filepath.is_file():
        df = pd.read_csv(points_filepath)
        points_geometry = [Point(xy) for xy in zip(df[args.longitude_column], df[args.latitude_column])]
        points_geo_df = gpd.GeoDataFrame(df, crs={'init': args.crs}, geometry=points_geometry)
        
        z_index = len(layers) + 1 if args.use_zordering else None
        points_geo_df.plot(ax=ax, markersize=args.point_size, marker='o', label=args.points_label, zorder=z_index)

        if args.partition_map:
            # Draw partition grid lines
            min_longitude, max_longitude = min(df[args.longitude_column]), max(df[args.longitude_column])
            longitude_stepsize = (max_longitude - min_longitude) / args.chunk_width
            for i in range(args.chunk_width + 1):
                longitude = min_longitude + longitude_stepsize * i
                plt.axvline(x=longitude, color='grey', linestyle='solid', linewidth=0.5)

            min_latitude, max_latitude = min(df[args.latitude_column]), max(df[args.latitude_column])
            latitude_stepsize = (max_latitude - min_latitude) / args.chunk_height
            for i in range(args.chunk_height + 1):
                latitude = min_latitude + latitude_stepsize * i
                plt.axhline(y=latitude, color='grey', linestyle='solid', linewidth=0.5)

            chunks = partition(points_geometry, args.chunk_width, args.chunk_height)
            if args.show_text:
                for x in range(args.chunk_width):
                    for y in range(args.chunk_height):
                        longitude = longitude_stepsize * x + 0.5 * longitude_stepsize + min_longitude
                        latitude = max_latitude - y * latitude_stepsize - 0.5 * latitude_stepsize
                        
                        count = len(chunks[x][y])
                        if args.count_adjustment_column is not None:
                            for point in chunks[x][y]:
                                adjustement_columns = df[(df[args.longitude_column] == point.x) & (df[args.latitude_column] == point.y)][args.count_adjustment_column]
                                count += sum(adjustement_columns)

                        text_extents = matplotlib.textpath.TextPath((0, 0), str(count), size=12).get_extents().transformed(ax.transData.inverted())
                        plt.text(longitude - 0.25 * text_extents.width, latitude - 0.25 * text_extents.height, \
                            '${{{}}}$'.format(count), size=12, zorder=1000)

plt.gca().set_title(args.title, pad=args.title_padding)
plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
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