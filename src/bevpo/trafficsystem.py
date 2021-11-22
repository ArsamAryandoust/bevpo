import bevpo.prob_dist as prob_dist
import bevpo.samp_traf as samp_traf
import bevpo.calc_tfsprop as calc_tfsprop
import bevpo.save_results as save_results

import math
import pandas as pd
import numpy as np

class TrafficSystem:

    """ Bundles all properties and methods of the traffic system
    modeled with bevpo.
    """

    def __init__(
        self,
        city_zone_coordinates,
        od_mean_travel_time_list,
        od_stddev_travel_time_list=None,
        od_distances=None,
        charging_profile=None,
        e_drive=2,
        e_dest=2,
        p_min=0.1,
        p_max=0.9,
        cars_per_zone=10
    ):

        ### Parameters
        self.city_zone_coordinates = city_zone_coordinates
        self.od_mean_travel_time_list = od_mean_travel_time_list
        self.od_stddev_travel_time_list = od_stddev_travel_time_list
        self.od_distances = od_distances
        self.charging_profile = charging_profile
        self.e_drive = e_drive
        self.e_dest = e_dest
        self.p_min = p_min
        self.p_max = p_max
        self.cars_per_zone = cars_per_zone
        
        ### Attributes
        self.T = len(od_mean_travel_time_list)
        self.number_zones = len(city_zone_coordinates)
        self.datatensor_mean = 0
        self.datatensor_stddev = 0
        self.C = round(
            self.number_zones * self.cars_per_zone
        )
        self.state_tensor = np.zeros(
            (
                self.C,
                self.T
            )
        ).astype(int)
        self.transition_tensor = np.zeros(
            (
                self.C,
                self.T,
                4 # driving x destination x travel time x distance
            )
        )
        # if no origin-destination travel distances passed,
        # calculate beeline distance between city zone centroids
        if od_distances is None:
            self.calc_od_distances(city_zone_coordinates)
        
        ### Results placeholders
        self.driving_map = 0
        self.parking_map = 0
        self.charging_map = 0
        self.avg_driving_times = 0
        self.avg_driving_distances = 0
        self.driving_share_lifetime = 0
        self.parking_share_lifetime = 0
        self.circadian_rhythm = 0
        self.distr_distances_per_t = 0
        self.distr_durations_per_t = 0
        self.distr_distances_total = 0
        self.distr_durations_total = 0
        self.distr_bins_km = 0
        self.distr_bins_s = 0
        self.charging_profile_dist = 0
        
        
    def calc_od_distances(self, city_zone_coordinates):
    
        """ Calculates the beeline distance between origin destination zones
        of a city if no explicit matrix of such distances was past when
        initializing class object.
        """
        
        c_lat_long = 111.3
        conv_deg_rad = 0.01745

        zone_id_array= city_zone_coordinates.index.values
        od_distances = pd.DataFrame(0, index=zone_id_array, columns=zone_id_array)

        # iterate over all city zones
        for entry_num1, zone_id1 in enumerate(zone_id_array):
            
            lat1 = city_zone_coordinates.loc[zone_id1]['zone_lat']
            long1 = city_zone_coordinates.loc[zone_id1]['zone_long']
            
            # iterate over all zones up to currently iterated one
            for entry_num2 in range(entry_num1+1):
                
                zone_id2 = zone_id_array[entry_num2]
                lat2 = city_zone_coordinates.loc[zone_id2]['zone_lat']
                long2 = city_zone_coordinates.loc[zone_id2]['zone_long']
                
                # calculate beeline distance
                distance_km = (
                    c_lat_long * math.sqrt(
                        (
                            math.cos(
                                (
                                    lat1
                                    + lat2
                                ) / 2 * conv_deg_rad
                            )
                        )**2 * (
                            long1
                            - long2
                        )**2
                        + (
                            lat1
                            - lat2
                        )**2
                    )
                )
                
                # write matrix entries symmetrically with 10% additional 
                # distance compared to the beeline
                od_distances.loc[zone_id1, zone_id2] = distance_km * 1.1
                od_distances.loc[zone_id2, zone_id1] = distance_km * 1.1
            
            # overwrite the diagonal entries with 1 km distances
            od_distances.loc[zone_id1, zone_id2] = 1
            od_distances.loc[zone_id2, zone_id1] = 1
            
        self.od_distances = od_distances
        

    def simulate_traffic(self):

        """ Simulates the traffic system when called. """
        
        ### Transform data from list of dataframes into single datatensor
        self.create_datatensors()
        
        ### Calculate distributions of driving and choosind a destination
        prob_dist.calc_prob_dists(self)
        
        ### Sample traffic
        samp_traf.sample_traffic(self)
        
        ### Calculate traffic system properties
        calc_tfsprop.calc_traffic_system_properties(self)
        

    def create_datatensors(self):

        """ Transforms the list of od matrices into a single datatensor. 
        City zone indexing corresponds to respective index positions in
        city_zone_coordinates.index. Time step indexing corresponds to 
        respective position of OD matrix in od_mean_travel_time_list.
        If od_stddev_travel_time_list is available, the same is done for this 
        list too.
        """

        datatensor_mean = np.zeros(
            (
                self.number_zones,
                self.number_zones,
                self.T
            )
        )
        # iterate over all time steps
        for time in range(self.T):
            # get od matrix of current time step
            od_mean_travel_time_df = self.od_mean_travel_time_list[time]

            # iterate over all rows in od matric
            for index, row in od_mean_travel_time_df.iterrows():
                # get the mean travel time column only
                mean = row['mean_travel_time']
                # set the source id to position where zone coordinate index matches
                source = int(
                    np.where(
                        self.city_zone_coordinates.index.values == row['source_id']
                    )[0]
                )
                # set also destination id to position where zone coordinate 
                # index matches
                dest = int(
                    np.where(
                        self.city_zone_coordinates.index.values == row['dest_id']
                    )[0]
                )
                # assign mean value to respective datatensor entry
                datatensor_mean[source, dest, time] = mean

        # set the list of od matrices to zero for saving memory
        self.od_mean_travel_time_list = 0
        # save extracted datatensor as class object attribute
        self.datatensor_mean = datatensor_mean

        # do the same for od standard deviations of travel time if available
        if self.od_stddev_travel_time_list is not None:
            datatensor_stddev = np.zeros(
                (
                    self.number_zones,
                    self.number_zones,
                    self.T
                )
            )

            for time in range(self.T):
                od_stddev_travel_time_df = self.od_stddev_travel_time_list[time]

                for index, row in od_stddev_travel_time_df.iterrows():
                    stddev = row['stddev_travel_time']
                    source = int(
                        np.where(
                            self.city_zone_coordinates.index.values == row['source_id']
                        )[0]
                    )
                    dest = int(
                        np.where(
                            self.city_zone_coordinates.index.values == row['dest_id']
                        )[0]
                    )

                datatensor_stddev[source, dest, time] = stddev

            self.od_stddev_travel_time_list = 0
            self.datatensor_stddev = datatensor_stddev
            
            
    def save_tfs_results(self, path_to_results=None):
    
        """ Saves the resulting plots and numeric values unter path_to_results
        when called.
        """
        
        # set path to current folder if no path is declared
        if path_to_results is None:
            path_to_results = './bevpo_results/'
            
        # set path as attribute of class object
        self.path_to_results = path_to_results
        
        # call function from bevpo.save_results.py module
        save_results.save_results(self)
