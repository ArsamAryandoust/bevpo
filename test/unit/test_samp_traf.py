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


class TestSampTraf(unittest.TestCase):

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
        
        print('Executed test_samp_traf.py')


    def setUp(self):

        """ Runs before every test. """

        pass


    def tearDown(self):

        """ Runs after every test. """

        pass


    def test_driving_activity_sampling(self):

        """ tests if driving activity sampling works by comparing changes
        in traffic system state and transition, before and after calling
        function.
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
            
            # create copy of driving matrix before sampling
            driving_matrix_before = tfs.transition_tensor[:, :, 0].copy()
            destination_matrix_before = tfs.transition_tensor[:, :, 1].copy()
            traveltime_matrix_before = tfs.transition_tensor[:, :, 2].copy()
            traveldistance_matrix_before = tfs.transition_tensor[:, :, 3].copy()
            
            # iterate over all time steps and sample traffic activity
            for t in range(tfs.T):
                samp_traf.driving_activity_sampling(
                    tfs,
                    t
                )
                
            # test if driving entries of transition matrix have changed 
            self.assertFalse(
                np.array_equal(
                    driving_matrix_before,
                    tfs.transition_tensor[:, :, 0]
                )
            )
            # test if travel time entries of transition matrix have changed 
            self.assertTrue(
                np.array_equal(
                    traveltime_matrix_before,
                    tfs.transition_tensor[:, :, 2]
                )
            )
            # test if travel distance entries of transition matrix have changed 
            self.assertTrue(
                np.array_equal(
                    traveldistance_matrix_before,
                    tfs.transition_tensor[:, :, 3]
                )
            )


    def test_destination_choice_sampling(self):

        """ tests if destination choice sampling works by comparing changes
        in traffic system state and transition, before and after calling
        function.
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
            
            # create p_drive and p_dest  
            prob_dist.calc_prob_dists(tfs)
            
            # create copy of driving matrix before sampling
            driving_matrix_before = tfs.transition_tensor[:, :, 0].copy()
            destination_matrix_before = tfs.transition_tensor[:, :, 1].copy()
            traveltime_matrix_before = tfs.transition_tensor[:, :, 2].copy()
            traveldistance_matrix_before = tfs.transition_tensor[:, :, 3].copy()
            
            # iterate over all time steps
            for t in range(tfs.T):
                # sample driving and choosing destination
                samp_traf.driving_activity_sampling(
                    tfs,
                    t
                )
                samp_traf.destination_choice_sampling(
                    tfs,
                    t
                )
                
            # test if destination entries of transition matrix have changed 
            self.assertFalse(
                np.array_equal(
                    destination_matrix_before,
                    tfs.transition_tensor[:, :, 1]
                )
            )
            # test if travel time entries of transition matrix have changed 
            self.assertTrue(
                np.array_equal(
                    traveltime_matrix_before,
                    tfs.transition_tensor[:, :, 2]
                )
            )
            # test if travel distance entries of transition matrix have changed 
            self.assertTrue(
                np.array_equal(
                    traveldistance_matrix_before,
                    tfs.transition_tensor[:, :, 3]
                )
            )


    def test_solve_initial_value_problem(self):

        """ tests if solving initial value problem works by comparing initial
        traffic system state before and after calling function.
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
            
            # create p_drive and p_dest 
            prob_dist.calc_prob_dists(tfs)
            
            # copy initial state before solving IVP
            initial_state_before = tfs.state_tensor[:, 0].astype(int)
            
            # solve initial value problem for traffic state
            samp_traf.solve_initial_value_problem(tfs)
            
            # test if driving entries of transition matrix have changed 
            self.assertFalse(
                np.array_equal(
                    initial_state_before,
                    tfs.state_tensor[:, 0].astype(int)
                )
            )
            
            
    def test_traveltime_and_distance_sampling(self):

        """ tests if sampling travel time and distance works by checking for
        negative entries in transition matrix.
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
            
            # create p_drive and p_dest  
            prob_dist.calc_prob_dists(tfs)
            
            # create copy of driving matrix before sampling
            driving_matrix_before = tfs.transition_tensor[:, :, 0].copy()
            destination_matrix_before = tfs.transition_tensor[:, :, 1].copy()
            traveltime_matrix_before = tfs.transition_tensor[:, :, 2].copy()
            traveldistance_matrix_before = tfs.transition_tensor[:, :, 3].copy()
            
            # iterate over all time steps
            for t in range(tfs.T):
                # sample driving and choosing destination
                samp_traf.driving_activity_sampling(
                    tfs,
                    t
                )
                samp_traf.destination_choice_sampling(
                    tfs,
                    t
                )
                samp_traf.traveltime_and_distance_sampling(
                    tfs,
                    t
                )
                
            # test if travel time entries of transition matrix have changed 
            self.assertFalse(
                np.array_equal(
                    traveltime_matrix_before,
                    tfs.transition_tensor[:, :, 2]
                )
            )
            # test if travel time entries are all positive
            self.assertTrue(
                (tfs.transition_tensor[:, :, 2] >= 0).all()
            )
            # test if travel distance entries are all positive
            self.assertTrue(
                (tfs.transition_tensor[:, :, 3] >= 0).all()
            )


    def test_sample_traffic(self):

        """ 
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
            
            # create p_drive and p_dest 
            prob_dist.calc_prob_dists(tfs)
            
            # copy last states before sampling traffic
            last_traffic_state_before = tfs.state_tensor[:, -1].astype(int).copy()
            last_transition_state_before = tfs.transition_tensor[:, -1, :].copy()
            
            # sample traffic
            samp_traf.sample_traffic(tfs)
            
            # test if traffic is sampled and states updated to last time step
            self.assertFalse(
                np.array_equal(
                    last_traffic_state_before,
                    tfs.state_tensor[:, -1].astype(int)
                )
            )
            self.assertFalse(
                np.array_equal(
                    last_transition_state_before,
                    tfs.transition_tensor[:, -1, :]
                )
            )

if __name__ == '__main__':

    unittest.main()
