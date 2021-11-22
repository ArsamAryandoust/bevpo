import unittest
import sys
sys.path.append('/bevpo/src')
import os

import bevpo.datasets.prep_ubermovement as prep_data


class TestPrepUbermovement(unittest.TestCase):

    """ Tests functions defined in datasets/prep_ubermovement.py """


    @classmethod
    def setUpClass(cls):

        """ Runs once before the first test. """

        # set path to data Uber Movement data
        path_to_data = 'data/public/Uber Movement/'
        
        # get list of cities
        city_list = os.listdir(path_to_data)
        
        # choose particular cities or comment out for testing all cities
        city_list = ['Perth']

        # set the ciy_list as attribute of unittest.TestCase
        cls.path_to_data = path_to_data
        cls.city_list = city_list
        

    @classmethod
    def tearDownClass(cls):

        """ Runs once after the last test. """
        
        print('Executed test_prep_ubermovement.py')


    def setUp(self):

        """ Runs before every test. """

        pass


    def tearDown(self):

        """ Runs after every test. """

        pass


    def test_import_geojson(self):
    
        """ Tests the geojson imports for every city available in the
        Uber Movement data folder
        """
        
        # iterate over all cities
        for city in self.city_list:
        
            # create the base path to data
            base_path = self.path_to_data + city + '/'
            file_list = os.listdir(base_path)

            # search directory for .json files
            json_file_name = [
                file for file in file_list if file.endswith('.json')
            ][0]

            # create the full paths to json and csv data
            path_to_json_data = base_path + json_file_name

            # process json file
            (
                map_movement_id_to_latitude_coordinates,
                map_movement_id_to_longitude_coordinates
            ) = prep_data.import_geojson(path_to_json_data)
            
            # test if there are more than one city zones.
            self.assertGreater(
                len(map_movement_id_to_latitude_coordinates),
                0
            )
            
            # test if number of list for lat and long coordinates match 
            self.assertEqual(
                len(map_movement_id_to_latitude_coordinates),
                len(map_movement_id_to_longitude_coordinates)
            )
            
            # iterate over all lists of lat and long coordinates
            for um_id, lat_l in map_movement_id_to_latitude_coordinates.items():
                long_l = map_movement_id_to_longitude_coordinates[um_id]
                
                # check if at least one coordinate is given for iterated zone
                self.assertGreater(
                    len(lat_l),
                    0
                )
                
                # check if number of lat and long coordinates match for each zone
                self.assertEqual(
                    len(lat_l),
                    len(long_l)
                )
                
                

    def test_calc_centroids(self):
    
        """ tests city zone centroid calculation for each city in the Uber
        Movement data folder.
        """
       
        # iterate over all cities
        for city in self.city_list:
        
            # create the base path to data
            base_path = self.path_to_data + city + '/'
            file_list = os.listdir(base_path)

            # search directory for .json files
            json_file_name = [
                file for file in file_list if file.endswith('.json')
            ][0]

            # create the full paths to json and csv data
            path_to_json_data = base_path + json_file_name

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
            
            # iterate over all city zone IDs
            for um_id, lat_centroid in map_movement_id_to_centroid_lat.items():
                long_centroid = map_movement_id_to_centroid_long[um_id]
                
                # get also list of coordinates for lat and long
                lat_l = map_movement_id_to_latitude_coordinates[um_id]
                long_l = map_movement_id_to_longitude_coordinates[um_id]
                
                # check if centroids are larger than smallest coordinate value
                self.assertGreater(
                    lat_centroid,
                    min(lat_l)
                )
                self.assertGreater(
                    long_centroid,
                    min(long_l)
                )
                
                # check if centroids are smaller than largest coordinate value
                self.assertLess(
                    lat_centroid,
                    max(lat_l)
                )
                self.assertLess(
                    long_centroid,
                    max(long_l)
                )
                

    def test_create_city_zone_coordinates(self):
    
        """ test for each Uber Movement city if created city zone coordinate
        matrices contain the correct columns, i.e. zone_lat and zone_long.
        """
        
        # iterate over all cities
        for city in self.city_list:
        
            # create the base path to data
            base_path = self.path_to_data + city + '/'
            file_list = os.listdir(base_path)

            # search directory for .json files
            json_file_name = [
                file for file in file_list if file.endswith('.json')
            ][0]

            # create the full paths to json and csv data
            path_to_json_data = base_path + json_file_name

            # merge into city_zone coordinates
            city_zone_coordinates = (
                prep_data.create_city_zone_coordinates(
                    path_to_json_data
                )
            )
            
            self.assertTrue(
                'zone_lat' in city_zone_coordinates.columns
            )
            self.assertTrue(
                'zone_long' in city_zone_coordinates.columns
            )
            self.assertGreater(
                len(city_zone_coordinates),
                0
            )
            
    def test_create_od_matrix_lists(self):
    
        """ test for each Uber Movement city if created city zone coordinate
        matrices contain the correct columns, i.e. zone_lat and zone_long.
        """

        # iterate over all cities
        for city in self.city_list:

            # create the base path to data
            base_path = self.path_to_data + city + '/'
            file_list = os.listdir(base_path)

            # search directory for .csv files
            csv_file_name = [
                file for file in file_list if file.endswith('.csv')
            ][0]

            # create the full paths to json and csv data
            path_to_rawdata = base_path + csv_file_name

            # merge into city_zone coordinates
            (
                od_mean_travel_time_list,
                od_std_travel_time_list
            ) = prep_data.create_od_matrix_lists(path_to_rawdata)
            
            # test if lists contain one matrix for each hour of day
            self.assertEqual(
                len(od_mean_travel_time_list),
                24
            )
            self.assertEqual(
                len(od_std_travel_time_list),
                24
            )
            
            # iterate over all matrices
            for mean_matrix, stddev_matrix in zip(
                od_mean_travel_time_list,
                od_std_travel_time_list
            ):
            
                self.assertTrue(
                    'source_id' in mean_matrix.columns
                )
                self.assertTrue(
                    'source_id' in stddev_matrix.columns
                )
                self.assertTrue(
                    'dest_id' in mean_matrix.columns
                )
                self.assertTrue(
                    'dest_id' in stddev_matrix.columns
                )
                self.assertTrue(
                    'mean_travel_time' in mean_matrix.columns
                )
                self.assertTrue(
                    'stddev_travel_time' in stddev_matrix.columns
                )

if __name__ == '__main__':

    unittest.main()
