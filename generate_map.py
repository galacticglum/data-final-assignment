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
parser.add_argument('inputs', type=str, nargs='+', help='The input shapefiles or layer descriptor files (JSON). Note: a layer descriptor file must have the JSON file extension.')
parser.add_argument('--title', type=str, help='The title of the graph.')
parser.add_argument('--colour', type=str, help='The colour of the plot.')
parser.add_argument('--xlabel', type=str, help='The label on the x-axis.', default='Latitude')
parser.add_argument('--ylabel', type=str, help='The label on the y-axis.', default='Longitude')
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
    layers[i].plot(ax=ax, zorder=i)

df = pd.read_csv('./geodata/red_light_camera_data.csv')
geometry = [Point(xy) for xy in zip(df['LONGITUDE'], df['LATITUDE'])]
geo_df = gpd.GeoDataFrame(df, crs={'init': 'WGS84'}, geometry=geometry)
geo_df.plot(ax=ax, markersize=20, marker='o', label='Red Light Camera', zorder=len(layers) + 1)

NUM_CHUNKS_X, NUM_CHUNKS_Y = 16, 10
min_longitude, max_longitude = min(df['LONGITUDE']), max(df['LONGITUDE'])
longitude_stepsize = (max_longitude - min_longitude) / NUM_CHUNKS_X
print(max_longitude, longitude_stepsize)
for i in range(NUM_CHUNKS_X + 1):
    longitude = min_longitude + longitude_stepsize * i
    plt.axvline(x=longitude, color='grey', linestyle='solid', linewidth=0.5)

min_latitude, max_latitude = min(df['LATITUDE']), max(df['LATITUDE'])
latitude_stepsize = (max_latitude - min_latitude) / NUM_CHUNKS_Y
for i in range(NUM_CHUNKS_Y + 1):
    latitude = min_latitude + latitude_stepsize * i
    plt.axhline(y=latitude, color='grey', linestyle='solid', linewidth=0.5)

# partition the points into chunks
chunks = [[[] for _ in range(NUM_CHUNKS_Y)] for _ in range(NUM_CHUNKS_X)]
for i in geometry:
    row = geo_df[geo_df['geometry'] == i]

    # Handle edge case for when a point lies on the
    # maximum longitudinal boundary or the minimum 
    # latitudinal bondary
    if i.x == max_longitude:
        chunk_x = NUM_CHUNKS_X - 1
    else:
        chunk_x = floor((i.x - min_longitude) / longitude_stepsize)

    if i.y == min_latitude:
        chunk_y = NUM_CHUNKS_Y - 1
    else:
        chunk_y = floor((max_latitude - i.y) / latitude_stepsize)

    chunks[chunk_x][chunk_y].append(row)

for x in range(NUM_CHUNKS_X):
    for y in range(NUM_CHUNKS_Y):
        longitude = longitude_stepsize * x + 0.5 * longitude_stepsize + min_longitude
        latitude = max_latitude - y * latitude_stepsize - 0.5 * latitude_stepsize
        
        text = str(len(chunks[x][y]))
        text_extents = matplotlib.textpath.TextPath((0, 0), text, size=12).get_extents().transformed(ax.transData.inverted())
        plt.text(longitude - 0.25 * text_extents.width, latitude - 0.25 * text_extents.height, text, size=12)

plt.title(args.title)
plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
plt.show()