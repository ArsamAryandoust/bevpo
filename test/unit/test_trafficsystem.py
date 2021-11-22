import unittest
import sys
sys.path.append('/bevpo/src')
import os
import random

import bevpo.datasets.prep_ubermovement as prep_data
import bevpo.trafficsystem as trafficsystem


class TestTrafficSystem(unittest.TestCase):

    """ Tests class methods defined in trafficsystem.py """


    @classmethod
    def setUpClass(cls):

        """ Runs once before the first test. """

        # set path to data Uber Movement data
        path_to_data = 'data/public/Uber Movement/'
        
        # get list of cities
        city_list = os.listdir(path_to_data)
        
        # choose particular cities or comment out for testing all cities
        city_list = ['Amsterdam']

        # set the ciy_list as attribute of unittest.TestCase
        cls.path_to_data = path_to_data
        cls.city_list = city_list
        

    @classmethod
    def tearDownClass(cls):

        """ Runs once after the last test. """
        
        print('Executed test_trafficsystem.py')


    def setUp(self):

        """ Runs before every test. """

        pass


    def tearDown(self):

        """ Runs after every test. """

        pass

    def test_calc_od_distances(self):
    
        """ tests if calc_od_distances method result in symmetric and possitive
        distance matrix.
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
                prep_data.create_city_zone_coordinates(path_to_json_data)
            )

            # create class instance wihtout od_distances to call 
            # method calc_od_distance
            tfs = trafficsystem.TrafficSystem(
                city_zone_coordinates,
                []
            )
            
            # check if all distances positive and greater zero
            for min_val in tfs.od_distances.min():
                  self.assertGreater(
                      min_val,
                      0
                  )
            
            # check if distance matrix is quadratic
            self.assertEqual(
                len(tfs.od_distances.index),
                len(tfs.od_distances.columns)
            )
            
            # sample 100 random datapoints from matrix
            dist_sample_1 = random.sample(
                list(tfs.od_distances.index),
                100
            )
            dist_sample_2 = random.sample(
                list(tfs.od_distances.index),
                100
            )
            
            for source_id, dest_id in zip(dist_sample_1, dist_sample_2):
            
                # check if matrix entries symmetric for randomly sampled points
                self.assertEqual(
                    tfs.od_distances.loc[source_id, dest_id],
                    tfs.od_distances.loc[dest_id, source_id]
                )
                
                # check if diagonal entries are set to 1 km.
                self.assertEqual(
                    tfs.od_distances.loc[source_id, source_id],
                    1
                )
            
    def test_create_datatensor(self):
    
        """ """

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
                prep_data.create_city_zone_coordinates(path_to_json_data)
            )

            # create class instance wihtout od_distances to call 
            # method calc_od_distance
            tfs = trafficsystem.TrafficSystem(
                city_zone_coordinates,
                []
            )
            
            # check if all distances positive and greater zero
            for min_val in tfs.od_distances.min():
                  self.assertGreater(
                      min_val,
                      0
                  )
            
            # check if distance matrix is quadratic
            self.assertEqual(
                len(tfs.od_distances.index),
                len(tfs.od_distances.columns)
            )
            
            # sample 100 random datapoints from matrix
            dist_sample_1 = random.sample(
                list(tfs.od_distances.index),
                100
            )
            dist_sample_2 = random.sample(
                list(tfs.od_distances.index),
                100
            )
            
            for source_id, dest_id in zip(dist_sample_1, dist_sample_2):
            
                # check if matrix entries symmetric for randomly sampled points
                self.assertEqual(
                    tfs.od_distances.loc[source_id, dest_id],
                    tfs.od_distances.loc[dest_id, source_id]
                )
                
                # check if diagonal entries are set to 1 km.
                self.assertEqual(
                    tfs.od_distances.loc[source_id, source_id],
                    1
                )
        

if __name__ == '__main__':

    unittest.main()
