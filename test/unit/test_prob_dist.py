import unittest
import sys
sys.path.append('/bevpo/src')
import os
import random
import numpy as np

import bevpo.datasets.prep_ubermovement as prep_data
import bevpo.trafficsystem as trafficsystem
import bevpo.prob_dist as prob_dist

class TestProbDist(unittest.TestCase):

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

        print('Executed test_prob_dist.py')


    def setUp(self):

        """ Runs before every test. """

        pass


    def tearDown(self):

        """ Runs after every test. """

        pass


    def test_create_distribution_p_drive(self):

        """ Tests if binomial probability distribution is created in the desired
        format.
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

            # initiate tfs class object
            tfs = trafficsystem.TrafficSystem(
                city_zone_coordinates,
                od_mean_travel_time_list,
                od_std_travel_time_list
            )
            tfs.create_datatensors()
            # create p_drive
            prob_dist.create_distribution_p_drive(tfs)
            
            # do tests
            self.assertEqual(
                    tfs.p_drive.shape[0],
                    tfs.number_zones
            )
            self.assertEqual(
                    tfs.p_drive.shape[1],
                    tfs.T
            )
            
            # test if each city zone has at least one non-zero binomial probability
            # of driving
            for entry in np.sum(tfs.p_drive, 0):
                self.assertGreater(
                        entry,
                        0
                )


    def test_create_distribution_p_dest(self):

        """ Tests if valid probability distribution is created by
        testing if distribution sums up to 1.
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

            # initiate tfs class object
            tfs = trafficsystem.TrafficSystem(
                city_zone_coordinates,
                od_mean_travel_time_list,
                od_std_travel_time_list
            )
            tfs.create_datatensors()
            
            # create p_dest
            prob_dist.create_distribution_p_dest(tfs)
            
            # test if all distributions sum up to one and are hence valid 
            for zone in range(tfs.number_zones):
                
                for t in range(tfs.T):
                    distribution = tfs.p_dest[zone, :, t]
                    if sum(distribution) != 0:
                        self.assertAlmostEqual(
                                sum(distribution),
                                1
                        )

    def test_create_distribution_p_joint(self):

        """ Tests if valid probability distribution is created 
        by comparing p_joint distributions over each source id 
        and time step to binomial probability of driving.
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

            # initiate tfs class object
            tfs = trafficsystem.TrafficSystem(
                city_zone_coordinates,
                od_mean_travel_time_list,
                od_std_travel_time_list
            )
            tfs.create_datatensors()
            
            # create p_drive, p_dest and p_joint
            prob_dist.create_distribution_p_drive(tfs)
            prob_dist.create_distribution_p_dest(tfs)
            prob_dist.create_distribution_p_joint(tfs)
            
            # test if all distributions sum up to the respective p_drive entry
            for zone in range(tfs.number_zones):
                
                for t in range(tfs.T):
                    distribution = tfs.p_joint[zone, :, t]
                    if sum(distribution) != 0:               
                        self.assertAlmostEqual(
                                sum(distribution),
                                tfs.p_drive[zone, t]
                        )
            
            
        
if __name__ == '__main__':

    unittest.main()
