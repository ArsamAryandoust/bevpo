import pandas as pd
import numpy as np
import math

def create_city_zone_coordinates(path_to_json_data):

    """ Calls the functions import_geojson and calc_centroids to get
    a list of city cone coordinates and created a pandas DataFrame in the 
    required format, i.e. wiht columns zone_lat and zone_long, as well as
    zone_id as index column
    """

    # map Uber Movement zone IDs to lists of lats and longs
    (
        map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    ) = import_geojson(path_to_json_data)

    # calculate centroids of city zone polygons
    (
        map_movement_id_to_centroid_lat,
        map_movement_id_to_centroid_long
    ) = calc_centroids(
       map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    )

    # create a pandas Dataframe in required format for bevpo
    city_zone_coordinates = pd.DataFrame(
        list(map_movement_id_to_centroid_lat.items()), 
        columns = ['zone_id','zone_lat']
    )
    
    city_zone_coordinates['zone_long'] = (
        map_movement_id_to_centroid_long.values()
    )
    
    # set the zone_id column as index
    city_zone_coordinates.set_index('zone_id', inplace=True)

    return city_zone_coordinates


def import_geojson(path_to_json_data):

    """ Imports the geojson data from the passed path and maps Uber Movement
    city zone IDs to a flattened list of latitude and longitude coordinates
    in the format of two dictionaries. Uses the recursive function called
    foster_coordinates_recursive to flatten the differently nested data.
    """
    
    data = pd.read_json(path_to_json_data)
    data.pop('type')
    data = data['features']
    
    map_json_entry_to_movement_id = dict()

    for json_id, json_entry in enumerate(data):
        
        map_json_entry_to_movement_id[json_id] = int(
          json_entry['properties']['MOVEMENT_ID']
        )
    
    map_movement_id_to_latitude_coordinates = dict()
    map_movement_id_to_longitude_coordinates = dict()

    for k, v in map_json_entry_to_movement_id.items():
        map_movement_id_to_latitude_coordinates[v] = []
        map_movement_id_to_longitude_coordinates[v] = []


    for json_id, movement_id in map_json_entry_to_movement_id.items():
        coordinates = data[json_id]['geometry']['coordinates']
        
        (
            map_movement_id_to_latitude_coordinates, 
            map_movement_id_to_longitude_coordinates
        ) = foster_coordinates_recursive(
            movement_id,
            map_movement_id_to_latitude_coordinates,
            map_movement_id_to_longitude_coordinates,
            coordinates
        )
        
    
    map_movement_id_to_coordinates = (
        map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    )

    return map_movement_id_to_coordinates


def foster_coordinates_recursive(
    movement_id,
    map_movement_id_to_latitude_coordinates,
    map_movement_id_to_longitude_coordinates,
    coordinates
):

    """ Flattens the coordinates of a passed city zone id (movement_id)
    and coordiates list recursively and saves their numeric values
    in the dictionaries that map movement ids to a list of latitude and 
    longitude coordinates.
    """

    dummy = 0

    for j in coordinates:

        if type(j) != list and dummy == 0:

            map_movement_id_to_longitude_coordinates[movement_id].append(j)
            dummy = 1
            continue

        elif type(j) != list and dummy == 1:

            map_movement_id_to_latitude_coordinates[movement_id].append(j)
            break

        else:

            dummy = 0
            coordinates = j
            (
                map_movement_id_to_latitude_coordinates,
                map_movement_id_to_longitude_coordinates
            ) = foster_coordinates_recursive(
                movement_id,
                map_movement_id_to_latitude_coordinates,
                map_movement_id_to_longitude_coordinates,
                coordinates
            )

    map_movement_id_to_coordinates = (
        map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    )

    return map_movement_id_to_coordinates


def calc_centroids(
    map_movement_id_to_latitude_coordinates,
    map_movement_id_to_longitude_coordinates
):

    """ Calculates the centroid of passed city zone polygons. Should a city
    zone consist of unregularities or multiple polygons, this is identified
    by centroid coordinates that are not within the bound of minimum and 
    maximum values of all coordinates of that city zone. In this case, the
    centroids are replaced with the mean of lat and long coordinates.
    """
    
    # create empty dictionary for mapping Uber Movement IDs to city zone areas
    map_movement_id_to_cityzone_area = dict()

    # iterate over all movement IDs and latitude coordinates
    for movement_id, lat_coordinates in map_movement_id_to_latitude_coordinates.items():
        
        # get also the longitude coordinates
        long_coordinates = map_movement_id_to_longitude_coordinates[movement_id]
        
        # calculate currently iterated city zone area
        area_cityzone = 0
        for i in range(len(lat_coordinates)-1):

            area_cityzone = (
                area_cityzone
                + long_coordinates[i] * lat_coordinates[i+1]
                - long_coordinates[i+1] * lat_coordinates[i]
            )
      
        area_cityzone = (
            area_cityzone
            + long_coordinates[i+1] * lat_coordinates[0]
            - long_coordinates[0] * lat_coordinates[i+1]
        )
        
        area_cityzone *= 0.5
        #area_cityzone = abs(area_cityzone)
        
        map_movement_id_to_cityzone_area[movement_id] = area_cityzone
        
    # create empty dictionaries for mapping Uber Movement IDs to city zone centroids
    map_movement_id_to_centroid_lat = dict()
    map_movement_id_to_centroid_long = dict()
        
    # iterate over all movement IDs and latitude coordinates
    for movement_id, lat_coordinates in map_movement_id_to_latitude_coordinates.items():
        
        # get also the longitude coordinates
        long_coordinates = map_movement_id_to_longitude_coordinates[movement_id]
        
        
        # calculate currently iterated city zone area
        centroid_lat = 0
        centroid_long = 0
        for i in range(len(lat_coordinates)-1):
            
            centroid_long += (
                long_coordinates[i]
                + long_coordinates[i+1]
            ) * (
                long_coordinates[i] * lat_coordinates[i+1]
                - long_coordinates[i+1] * lat_coordinates[i]
            )

            centroid_lat += (
                lat_coordinates[i]
                + lat_coordinates[i+1]
            ) * (
                long_coordinates[i] * lat_coordinates[i+1]
                - long_coordinates[i+1] * lat_coordinates[i]
            )

        centroid_long += (
            long_coordinates[i+1]
            + long_coordinates[0]
        ) * (
            long_coordinates[i+1] * lat_coordinates[0]
            - long_coordinates[0] * lat_coordinates[i+1]
        )
        
        centroid_lat += (
                lat_coordinates[i+1]
                + lat_coordinates[0]
            ) * (
                long_coordinates[i+1] * lat_coordinates[0]
                - long_coordinates[0] * lat_coordinates[i+1]
            )
        

        centroid_lat /= (
            6 * map_movement_id_to_cityzone_area[movement_id]
        )
        centroid_long /= (
            6 * map_movement_id_to_cityzone_area[movement_id]
        )
     
        # Uber Movement city zones sometimes consist of multiple distinct polygons
        if (
            centroid_lat < min(lat_coordinates)
            or centroid_lat > max(lat_coordinates)
            or centroid_long < min(long_coordinates)
            or centroid_long > max(long_coordinates)
        ):
            # in this case we calculate the mean instead of centroid
            centroid_lat = np.mean(lat_coordinates)
            centroid_long = np.mean(long_coordinates)            
        
        map_movement_id_to_centroid_lat[movement_id] = centroid_lat
        map_movement_id_to_centroid_long[movement_id] = centroid_long
        
    map_movement_id_to_centroid_coordinates = (
        map_movement_id_to_centroid_lat,
        map_movement_id_to_centroid_long
    )
    
    return map_movement_id_to_centroid_coordinates
    

def create_od_matrix_lists(path_to_rawdata):

    """ imports the raw Uber Movement travel time data and creates tw0 lists of
    origin destination (OD) matrices. A first list contains the mean travel time
    and a distinct second list contains the standard deviation of travel time. 
    Each matrix and therefore each list element corresponds to a single time 
    step. 
    """
    
    rename_dict_mean = {
        'sourceid': 'source_id',
        'dstid': 'dest_id'
    }
    
    rename_dict_stddev = {
        'sourceid': 'source_id',
        'dstid': 'dest_id',
        'standard_deviation_travel_time': 'stddev_travel_time'
    }
    
    travel_data = pd.read_csv(path_to_rawdata)
    od_mean_travel_time_list = []
    od_std_travel_time_list = []

    T = 24
    for t in range(T):
        hod_travel_data = travel_data.loc[travel_data['hod'] == t]
        od_mean_travel_time_list.append(
            hod_travel_data[
                ['sourceid', 'dstid','mean_travel_time']
            ].rename(columns=rename_dict_mean)
        )
        od_std_travel_time_list.append(
            hod_travel_data[
                ['sourceid', 'dstid','standard_deviation_travel_time']
            ].rename(columns=rename_dict_stddev)
        )
    
    
    od_travel_time_lists = (
        od_mean_travel_time_list,
        od_std_travel_time_list
    )
    
    return od_travel_time_lists
