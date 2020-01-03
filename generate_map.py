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
from math import floor
from pathlib import Path
from utils import init_logger
from shapely.geometry import Point, Polygon

logger = init_logger()
parser = argparse.ArgumentParser(description='Generate a map given a set of shapefiles.')
parser.add_argument('inputs', type=str, nargs='+', help='The input shapefiles or layer descriptor files (JSON). ' +
                                                        'Note: a layer descriptor file must have the JSON file extension.')
parser.add_argument('--points', dest='points_filepath', type=str, help='The path to the CSV file containing the geodata of points on a map.')
parser.add_argument('--points-label', type=str, help='The label of the geodata points.')
parser.add_argument('--longitude-col', type=str, help='The (case-sensitive) header name of the longitude column.', default='LONGITUDE')
parser.add_argument('--latitude-col', type=str, help='The (case-sensitive) header name of the latitude column.', default='LATITUDE')
parser.add_argument('--crs', type=str, help='The coordinate reference system of the geodata.', default='WGS84')
parser.add_argument('--disable-zordering', dest='use_zordering', help='Order the layers in ascending order (order: map layers and then point layer).', action='store_false')
parser.add_argument('--title', type=str, help='The title of the graph.')
parser.add_argument('--colour', type=str, help='The colour of the plot.')
parser.add_argument('--xlabel', type=str, help='The label on the x-axis.', default='Latitude')
parser.add_argument('--ylabel', type=str, help='The label on the y-axis.', default='Longitude')
parser.add_argument('--partition', dest='partition_map', help='Partition the data points on the map into chunks.', action='store_true')
parser.add_argument('-cw', '--chunk-width', type=int, help='The number of chunks on the longitude (horizontal). Defaults to 32.', default=32)
parser.add_argument('-ch', '--chunk-height', type=int, help='The number of chunks on the latitude (vertical). Defaults to 32.', default=32)
parser.set_defaults(partition=False, use_zordering=True)
args = parser.parse_args()

if len(args.inputs) == 0:
    logger.error('No shapefile inputs were specified.')
    exit(1)

class Layer:
    def __init__(self, shapefile, linewidth=None, colour=None, **kwargs):
        self.shapefile = shapefile
        self.linewidth = linewidth
        self.colour = colour or args.colour

    def plot(self, **kwargs):
        self.shapefile.plot(linewidth=self.linewidth, color=self.colour, **kwargs)

def verify_shapefile_path(filepath):
    filepath = Path(filepath)
    if not (filepath.exists() or filepath.is_file()):
        logger.warn('The specified shapefile input (\'{}\') is not a file or does not exist! Skipping this shapefile.'.format(str(filepath)))
        return False

    return True
def load_layer_description(filepath):
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

layers = []
for input_pattern in args.inputs:
    files = glob.glob(input_pattern, recursive=True)
    for input_path in files:
        filepath = Path(input_path)
        if filepath.suffix == '.json':
            layers.extend(load_layer_description(filepath))
        else:
            if not verify_shapefile_path(filepath): continue
            shapefile = gpd.read_file(filepath)
            layers.append(Layer(shapefile))
 
fig, ax = plt.subplots()
for i in range(len(layers)):
    z_index = i if args.use_zordering else None
    layers[i].plot(ax=ax, zorder=z_index)
    
if args.points_filepath is not None:
    points_filepath = Path(args.points_filepath).resolve().absolute()
    if points_filepath.exists() and points_filepath.is_file():
        df = pd.read_csv(points_filepath)
        points_geometry = [Point(xy) for xy in zip(df[args.longitude_col], df[args.latitude_col])]
        points_geo_df = gpd.GeoDataFrame(df, crs={'init': args.crs}, geometry=points_geometry)
        
        z_index = len(layers) + 1 if args.use_zordering else None
        points_geo_df.plot(ax=ax, markersize=20, marker='o', label=args.points_label, zorder=z_index)

        if args.partition_map:
            min_longitude, max_longitude = min(df[args.longitude_col]), max(df[args.longitude_col])
            longitude_stepsize = (max_longitude - min_longitude) / args.chunk_width
            for i in range(args.chunk_width + 1):
                longitude = min_longitude + longitude_stepsize * i
                plt.axvline(x=longitude, color='grey', linestyle='solid', linewidth=0.5)

            min_latitude, max_latitude = min(df[args.latitude_col]), max(df[args.latitude_col])
            latitude_stepsize = (max_latitude - min_latitude) / args.chunk_height
            for i in range(args.chunk_height + 1):
                latitude = min_latitude + latitude_stepsize * i
                plt.axhline(y=latitude, color='grey', linestyle='solid', linewidth=0.5)

            # partition the points into chunks
            chunks = [[[] for _ in range(args.chunk_height)] for _ in range(args.chunk_width)]
            for point in points_geometry:
                row = points_geo_df[points_geo_df['geometry'] == point]

                # Handle edge case for when a point lies on the
                # maximum longitudinal boundary or the minimum 
                # latitudinal bondary
                if point.x == max_longitude:
                    chunk_x = args.chunk_width - 1
                else:
                    chunk_x = floor((point.x - min_longitude) / longitude_stepsize)

                if point.y == min_latitude:
                    chunk_y = args.chunk_height - 1
                else:
                    chunk_y = floor((max_latitude - point.y) / latitude_stepsize)

                chunks[chunk_x][chunk_y].append(row)

            for x in range(args.chunk_width):
                for y in range(args.chunk_height):
                    longitude = longitude_stepsize * x + 0.5 * longitude_stepsize + min_longitude
                    latitude = max_latitude - y * latitude_stepsize - 0.5 * latitude_stepsize
                    
                    text = str(len(chunks[x][y]))
                    text_extents = matplotlib.textpath.TextPath((0, 0), text, size=12).get_extents().transformed(ax.transData.inverted())
                    plt.text(longitude - 0.25 * text_extents.width, latitude - 0.25 * text_extents.height, text, size=12)

plt.title(args.title)
plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
plt.show()