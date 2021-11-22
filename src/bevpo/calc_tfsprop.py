import numpy as np
import matplotlib.pyplot as plt
import imageio


def calc_traffic_system_properties(tfs):

    """ Calculates a series of traffic system properties resulting from the 
    sampling. 
    """
    
    create_parking_and_driving_maps(tfs)
    calc_traffic_properties(tfs)
    calc_travel_distributions(tfs)
    calc_charging_distributions(tfs)
    

def calc_charging_distributions(tfs):

    """ Calculates the distribution of charging over time and space by splitting
    the passed charging profile over parked cars in each time step. CAUTION 
    ASSUMPTION: parked cars are charged at every given time.
    """
    
    if tfs.charging_profile is not None:
    
        if tfs.T != len(tfs.charging_profile):
        
            print(
                'Provided charging profile does not match number of time steps'
            )
            
        else:

            # turn charging profile into a numpy array
            charging_profile = np.array(tfs.charging_profile)
            
            # min-max scale provided charging_profile
            charging_profile_minmax = (
                charging_profile - min(charging_profile) / (
                    max(charging_profile) - min(charging_profile)
                )
            )
            
            # scale to valid distribution that sums to 1
            charging_profile_dist = (
                charging_profile / sum(charging_profile)
            )
            
            # copy the distribution of parking cars
            tfs.charging_map = tfs.parking_map.copy()
            
            # iterate over timesteps and scale parking cars to charging profile
            for t in range(tfs.T):
                tfs.charging_map[:, t] /= charging_profile_dist[t]
                
            # scale distribution to unit sum 
            tfs.charging_map /= np.sum(tfs.charging_map)
            
            # save the scaled charging profile
            tfs.charging_profile_dist = charging_profile_dist
    

def create_parking_and_driving_maps(tfs):

    """ Normalizes state and state transition matrices and produces
    the resulting parking and driving maps.
    """

    driving_map = np.zeros(
        (
            tfs.number_zones, 
            tfs.T
        )
    )
    parking_map = np.zeros(
        (
            tfs.number_zones, 
            tfs.T
        )
    )
    for car in range(tfs.C):
    
        for t in range(tfs.T):
            zone = tfs.state_tensor[car, t]
            driving_activity = tfs.transition_tensor[car, t, 0]

            if driving_activity == 1:
                driving_map[zone-1, t] += 1
            else:
                parking_map[zone-1, t] += 1

    tfs.driving_map = driving_map / np.sum(driving_map)
    tfs.parking_map = parking_map / np.sum(parking_map)


def calc_traffic_properties(tfs):

    """ Calculates statistics on average driving times and distances and driving 
    and parking shares per lifetime, as well as circadian rhythm of traffic. 
    """
    
    # two values for driving time, distance
    avg_properties = np.zeros(
        (
            tfs.T+1,
            2
        )
    )
    for t in range(tfs.T):
        driving_time_t = tfs.transition_tensor[:, t, 2] / 60
        driving_distance_t = tfs.transition_tensor[:, t, 3]
        avg_properties[t+1, 0] = np.mean(driving_time_t)
        avg_properties[t+1, 1] = np.mean(driving_distance_t)
        
    avg_properties[0, 0] = np.mean(avg_properties[1:, 0])
    avg_properties[0, 1] = np.mean(avg_properties[1:, 1])
    
    # calculate driving and parking shares per lifetime
    driving_share_lifetime = round(
        avg_properties[0, 0] / 60 * 100
    )
    parking_share_lifetime = 100 - driving_share_lifetime
    
    
    ### Calculuate circadian rhythm
    circadian_rhythm = np.sum(
        tfs.driving_map,
        axis=0
    )
    min_traf = np.min(circadian_rhythm)
    max_traf = np.max(circadian_rhythm)
    circadian_rhythm = (circadian_rhythm - min_traf) / (max_traf - min_traf)

    ### save results to class object attributes
    tfs.avg_driving_times = avg_properties[:, 0]
    tfs.avg_driving_distances = avg_properties[:, 1]
    tfs.driving_share_lifetime = driving_share_lifetime
    tfs.parking_share_lifetime = parking_share_lifetime
    tfs.circadian_rhythm = circadian_rhythm


def calc_travel_distributions(tfs):
    
    """ calculates the multivariate distribution of travelled distances and 
    durations for both fine-grained time steps and all time steps together.
    """
    
    # calculate overall distributions
    duration_histogram = np.histogram(
        tfs.transition_tensor[:, :, 2]
    )
    distance_histogram = np.histogram(
        tfs.transition_tensor[:, :, 3]
    )
    bins_km = distance_histogram[1][1:]
    bins_s = duration_histogram[1][1:]
    distr_durations_total = np.round(
        duration_histogram[0] / sum(duration_histogram[0]) * 100,
        2
    )
    distr_distances_total = np.round(
        distance_histogram[0] / sum(distance_histogram[0]) * 100,
        2
    )
    distr_durations_per_t = np.zeros(
        (
            tfs.T,
            len(bins_s)
        )
    )
    distr_distances_per_t = np.zeros(
        (
            tfs.T,
            len(bins_km)
        )
    )
    
    # calculate distributions over each time step
    for t in range(tfs.T):
        durations_t = tfs.transition_tensor[:, t, 2]
        distances_t = tfs.transition_tensor[:, t, 3]
        distr_durations_t = np.histogram(
            durations_t,
            len(bins_s),
            (
                bins_s[0],
                bins_s[-1]
            )
        )[0]
        distr_distances_t = np.histogram(
            distances_t,
            len(bins_km),
            (
                bins_km[0],
                bins_km[-1]
            )
        )[0]
        
        distr_durations_per_t[t, :] = distr_durations_t
        distr_distances_per_t[t, :] = distr_distances_t
        
    # assign as class properties
    tfs.distr_distances_per_t = distr_distances_per_t
    tfs.distr_durations_per_t = distr_durations_per_t
    tfs.distr_distances_total = distr_distances_total
    tfs.distr_durations_total = distr_durations_total
    tfs.distr_bins_km = bins_km
    tfs.distr_bins_s = bins_s
    
      
        
