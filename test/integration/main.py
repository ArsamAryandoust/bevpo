### Import packages
import sys
sys.path.append('/bevpo/src')
import os
import time

import bevpo.datasets.prep_ubermovement as prep_data
import bevpo.trafficsystem as trafficsystem
import bevpo.samp_traf as samp_traf

# get starting time
start_t = time.perf_counter()


### Set some paths

# set path to Uber Movement data
path_to_data = 'data/public/Uber Movement/'

# get list of cities
city_list = os.listdir(path_to_data)

# choose particular cities or comment out for testing all cities
city_list = ['Amsterdam']

for city in city_list:

    ### Prepare Uber Data 

    # create the base path to data
    base_path = path_to_data + city + '/'
    file_list = os.listdir(base_path)

    # search directory for .json files
    json_file_name = [file for file in file_list if file.endswith('.json')][0]

    # search directory for .csv files
    csv_file_name = [file for file in file_list if file.endswith('.csv')][0]

    # create the full paths to json and csv data
    path_to_json_data = base_path + json_file_name
    path_to_rawdata = base_path + csv_file_name

    # process json file
    (
        map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    ) = prep_data.import_geojson(path_to_json_data)

    # prepare city zone centroids
    (
        map_movement_id_to_centroid_lat,
        map_movement_id_to_centroid_long
    ) = prep_data.calc_centroids(
        map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    )

    # merge into city_zone coordinates
    city_zone_coordinates = (
        prep_data.create_city_zone_coordinates(path_to_json_data)
    )

    # create list of OD travel time matrices
    (
        od_mean_travel_time_list,
        od_std_travel_time_list
    ) = prep_data.create_od_matrix_lists(path_to_rawdata)


    ### Simulate traffic
    
    # create a charging profile of length 24
    charging_profile = [
        5, 6, 7, 10, 10, 9, 8, 7, 6, 3, 2, 1, 5, 7, 4, 2, 1, 6, 4, 5, 5, 3, 7, 6
    ]

    # initiate without od_std_travel_time_list
    tfs = trafficsystem.TrafficSystem(
        city_zone_coordinates,
        od_mean_travel_time_list,
        charging_profile=charging_profile
    )

    # test without od_std_travel_time_list
    tfs.create_datatensors()

    # tfs.create_datatensor() removes od_travel_time_lists, initiate new class object
    tfs = trafficsystem.TrafficSystem(
        city_zone_coordinates,
        od_mean_travel_time_list,
        od_std_travel_time_list,
        charging_profile=charging_profile
    )

    # test now with od_std_travel_time_list, calls again tfs.create_datatensor()
    tfs.simulate_traffic()
    tfs.save_tfs_results()

# get ending time
end_t = time.perf_counter()

# tell us how long integration test needed
print(f'Executed main.py in {round(end_t-start_t, 2)} second(s)')
