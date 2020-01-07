'''
Maps the automobile crash data to the intersection data.
'''

import pandas as pd

intersection_df = pd.read_csv('geodata/intersection_data.csv')
crash_df = pd.read_csv('geodata/automobile_crash_data.csv')

# Clean the data
def clean_street_name(x):
    # Remove apostrophes from names
    x = x.replace('\'', ' ')

    # Special cases
    if 'HIGHWAY 27' in x:
        x = 'HWY 27'
    
    return x.strip().upper()

new_df = intersection_df.copy()
intersection_df['LINEAR_NAME_FULL_1'] = intersection_df['LINEAR_NAME_FULL_1'].apply(clean_street_name)
intersection_df['LINEAR_NAME_FULL_2'] = intersection_df['LINEAR_NAME_FULL_2'].apply(clean_street_name)
crash_df['STREET1'] = crash_df['STREET1'].apply(clean_street_name)
crash_df['STREET2'] = crash_df['STREET2'].apply(clean_street_name)

# A dictionary of crashes by year
non_fatal_crashes_by_year = {}
fatal_crashes_by_year = {}
for year in range(2007, 2019):
    non_fatal_crashes_by_year[year] = []
    fatal_crashes_by_year[year] = []

for index, row in intersection_df.iterrows():
    street_name_condition = ((crash_df['STREET1'] == row['LINEAR_NAME_FULL_1']) & \
                                (crash_df['STREET2'] == row['LINEAR_NAME_FULL_2'])) | \
                             ((crash_df['STREET1'] == row['LINEAR_NAME_FULL_2']) & \
                                (crash_df['STREET2'] == row['LINEAR_NAME_FULL_1']))

    crashes = crash_df[street_name_condition & (crash_df['LOCCOORD'] == 'Intersection')]
    def count_crashes(accident_class):
        result = {}
        for year in range(2007, 2019):
            result[year] = len(crashes[(crashes['YEAR'] == year) & (crashes['ACCLASS'] == accident_class)])

        return result

    non_fatal_crashes = count_crashes('Non-Fatal Injury')
    for year in non_fatal_crashes:
        non_fatal_crashes_by_year[year].append(non_fatal_crashes[year])

    fatal_crashes = count_crashes('Fatal')
    for year in fatal_crashes:
        fatal_crashes_by_year[year].append(fatal_crashes[year])

# Rename columns
non_fatal_crashes_keys = list(non_fatal_crashes_by_year.keys())
for year in non_fatal_crashes_keys:
    non_fatal_crashes_by_year['{}_NON_FATAL'.format(year)] = non_fatal_crashes_by_year.pop(year)
fatal_crashes_keys = list(fatal_crashes_by_year.keys())
for year in fatal_crashes_keys:
    fatal_crashes_by_year['{}_FATAL'.format(year)] = fatal_crashes_by_year.pop(year)

new_df = new_df.assign(**non_fatal_crashes_by_year, **fatal_crashes_by_year)
new_df.to_csv('geodata/intersection_data_crashes.csv', index=False)