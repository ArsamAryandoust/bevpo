import unittest
import sys
sys.path.append('/bevpo/src')
import os
import random
import numpy as np

import bevpo.datasets.prep_ubermovement as prep_data
import bevpo.trafficsystem as trafficsystem
import bevpo.prob_dist as prob_dist
import bevpo.samp_traf as samp_traf
import bevpo.calc_tfsprop as calc_tfsprop


class TestPrepUbermovement(unittest.TestCase):

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
        
        print('Executed test_calc_tfsprop.py')

    def setUp(self):

        """ Runs before every test. """

        pass


    def tearDown(self):

        """ Runs after every test. """

        pass


    def test_calc_traffic_system_properties(self):

        """ tests if distributions of parking and driving 
        cars sum up to 1.
        """
        
        for city in self.city_list:

            ### 1. Prepare Uber Data 
            
            # create the base path to data
            base_path = self.path_to_data + city + '/'
            file_list = os.listdir(base_path)
            
            # search directory for .json files
            json_file_name = [
                file for file in file_list if file.endswith('.json')
            ][0]
            
            # search directory for .csv files
            csv_file_name = [
                file for file in file_list if file.endswith('.csv')
            ][0]
            
            # create the full paths to json and csv data
            path_to_json_data = base_path + json_file_name
            path_to_rawdata = base_path + csv_file_name
            
            # merge into city_zone coordinates
            city_zone_coordinates = (
                prep_data.create_city_zone_coordinates(path_to_json_data)
            )
            
            # create list of OD travel time matrices
            (
                od_mean_travel_time_list,
                od_std_travel_time_list
            ) = prep_data.create_od_matrix_lists(path_to_rawdata)
            
            
            ### 2. Simulate traffic 
            
            # create a charging profile of length 24
            charging_profile = [
                5, 6, 7, 10, 10, 9, 8, 7, 6, 3, 2, 1, 5, 7, 4, 2, 1, 6, 
                4, 5, 5, 3, 7, 6
            ]
            
            # initiate tfs class object
            tfs = trafficsystem.TrafficSystem(
                city_zone_coordinates,
                od_mean_travel_time_list,
                od_std_travel_time_list,
                charging_profile=charging_profile
            )
            tfs.create_datatensors()
            
            # create p_drive and p_dest 
            prob_dist.calc_prob_dists(tfs)
            
            # sample traffic
            samp_traf.sample_traffic(tfs)
            
            # calculate properties.
            calc_tfsprop.calc_traffic_system_properties(tfs)
            
            ### Test generated maps
            
            # test if distributions of parking cars sum up to 1
            self.assertAlmostEqual(
                np.sum(tfs.parking_map), 
                1
            )
            
            # test if distributions of driving cars sum up to 1
            self.assertAlmostEqual(
                np.sum(tfs.driving_map), 
                1
            )
            
            # test if distributions of charging cars sum up to 1
            self.assertAlmostEqual(
                np.sum(tfs.charging_map), 
                1
            )
            
            
if __name__ == '__main__':

    unittest.main()
