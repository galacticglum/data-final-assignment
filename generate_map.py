'''
Generate a map given a set of shapefiles.
'''

import json
import glob
import argparse
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Point, Polygon
from utils import init_logger

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
geo_df.plot(ax=ax, markersize=20, color='red', marker='o', label='Red Light Camera', zorder=len(layers) + 1)

plt.title(args.title)
plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
plt.show()