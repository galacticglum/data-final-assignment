'''
Maps the traffic volume to the intersection data.
'''

import pandas as pd

charges_df = pd.read_csv('geodata/red_light_camera_charges.new.csv')
volume_df = pd.read_csv('geodata/traffic_volumes.csv',  keep_default_na=False)

# Clean the data
def clean_street_name(x):
    # Remove apostrophes from names
    x = x.replace('\'', ' ')
    return x.strip().upper()

new_df = charges_df.copy()
charges_df['LINEAR_NAME_FULL_1'] = charges_df['LINEAR_NAME_FULL_1'].apply(clean_street_name)
charges_df['LINEAR_NAME_FULL_2'] = charges_df['LINEAR_NAME_FULL_2'].apply(clean_street_name)
volume_df['Main'] = volume_df['Main'].apply(clean_street_name)
volume_df['Side 1 Route'] = volume_df['Side 1 Route'].apply(clean_street_name) 

volume_data = {
    'DATE': [],
    'VEHICLE_VOLUME': [],
    'PEDESTRIAN_VOLUME': []
}

for index, row in charges_df.iterrows():
    # Row corresponding to the intersection on the volume data
    intersection_row = volume_df[(volume_df['Main'] == row['LINEAR_NAME_FULL_1']) \
        & (volume_df['Side 1 Route'] == row['LINEAR_NAME_FULL_2']) \
        & (volume_df['Midblock Route'] == '')]
    
    date, vehicle_volume, pedestrian_volume = '', '', ''
    if not intersection_row.empty:
        intersection_row = intersection_row.iloc[0]
        date = intersection_row['Count Date']
        vehicle_volume = intersection_row['8 Peak Hr Vehicle Volume']
        pedestrian_volume = intersection_row['8 Peak Hr Pedestrian Volume']
    
    volume_data['DATE'].append(date)
    volume_data['VEHICLE_VOLUME'].append(vehicle_volume)
    volume_data['PEDESTRIAN_VOLUME'].append(pedestrian_volume)

# print(volume_data)
new_df = new_df.assign(**volume_data)

# Rename
new_df = new_df.rename(columns={
    'DATE': 'Date', 
    'VEHICLE_VOLUME': '8 Peak Hr Vehicle Volume', 
    'PEDESTRIAN_VOLUME': '8 Peak Hr Pedestrian Volume'
})

new_df.to_csv('geodata/intersection_data.csv', index=False)