import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

trn_motorway = gpd.read_file('./geodata/trn_export/trn_motorway.shp')
trn_primary = gpd.read_file('./geodata/trn_export/trn_primary.shp')
trn_secondary = gpd.read_file('./geodata/trn_export/trn_secondary.shp')
trn_tertiary = gpd.read_file('./geodata/trn_export/trn_tertiary.shp')

fig, ax = plt.subplots()
trn_motorway.plot(linewidth=0.6, ax=ax)
trn_primary.plot(linewidth=0.5, ax=ax)
trn_secondary.plot(linewidth=0.3, ax=ax)
trn_tertiary.plot(linewidth=0.26, ax=ax)

plt.title('Toronto Road Network')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()