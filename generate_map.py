import geopandas as gpd

shapefile = 'instance/TOPO_EDGE_OF_ROAD_WGS84.shp'

# Read shapefile using Geopandas
gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]

# Rename columns.
gdf.columns = ['country', 'country_code', 'geometry']
gdf.head()