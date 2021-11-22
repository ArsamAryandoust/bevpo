import numpy as np
import random


def sample_traffic(tfs):

    """ Initializes the traffic system state and then samples traffic for
    all time steps.
    """

    # solve initial value problem for traffic state
    solve_initial_value_problem(tfs)

    # simluate over all time steps
    for t in range(tfs.T):
        driving_activity_sampling(
            tfs,
            t
        )
        destination_choice_sampling(
            tfs,
            t
        )
        traveltime_and_distance_sampling(
            tfs,
            t
        )
        # update state matrix only up to last time step,
        # but sample transition matrix one step beyond last.
        if t < tfs.T-1:
            tfs.state_tensor[:, t+1] = tfs.transition_tensor[:, t, 1]

    # set distributions to zero for saving memory
    tfs.p_drive = 0
    tfs.p_dest = 0
    tfs.p_joint = 0

def solve_initial_value_problem(tfs):

    """ Initializes the traffic system state by first distributing all cars
    uniformly accross all city zones and then sampling through all time steps.
    The resulting system state after an entire sampling process is then taken
    as the initial system state. """

    # uniformly assign cars to to all city zones for state in initial time step
    zone = 0
    for car in range(0, tfs.C, tfs.cars_per_zone):
        tfs.state_tensor[car:(car+tfs.cars_per_zone), 0] = zone
        zone += 1
        
    # transform to integer values
    tfs.state_tensor = tfs.state_tensor.astype(int)
    
    # sample over all time steps
    for t in range(tfs.T):
        driving_activity_sampling(
            tfs,
            t
        )
        destination_choice_sampling(
            tfs,
            t
        )

        if t < tfs.T-1:
            tfs.state_tensor[:, t+1] = tfs.transition_tensor[:, t, 1]
        else:
            tfs.state_tensor[:, 0] = tfs.transition_tensor[:, -1, 1].astype(int)


def driving_activity_sampling(
    tfs,
    t
):

    """ Samples if a car drives or stays parked in a respective city zone
    from p_drive.
     """

    # iterate over all cars we simulate
    for car in range(tfs.C):
        # set origin zone to location of car in t
        origin = tfs.state_tensor[car, t]
        # get driving probability for origin-time combination
        driving_probability = tfs.p_drive[origin, t]
        # sample if car drives from this probability
        drive = np.random.choice(
            [1, 0],
            p=[driving_probability, 1-driving_probability]
        )
        # assig sampling outcome to transition matrix of car-timestep combination
        tfs.transition_tensor[car, t, 0] = drive
        # if car stays parked, also assign destination zone to stay orgin
        # in the transition matrix of car-timestep combination
        if drive == 0:
            tfs.transition_tensor[car, t, 1] = origin


def destination_choice_sampling(
    tfs,
    t
):

    """ Samples travel destinations from p_dest. """

    # iterate over all cars
    for car in range(tfs.C):
        
        # get binary sampling result from driving_activity_sampling() 
        drive = tfs.transition_tensor[car, t, 0]
        
        # only if car is sampled to drive, make a change to transition matrix
        # otherwise, we have already assigned destination to stay origin of 
        # car in in tfs.transition_tensor during driving_activity_sampling().
        if drive == 1:
            # get location of car
            origin = tfs.state_tensor[car, t]
            # get probability distribution of location-timestep combination
            distribution = tfs.p_dest[origin, :, t]
            # if distribution sum is zero, car has destination in same zone
            if sum(distribution) == 0:
                destination = origin
            else:
                # otherwise, car chooses destination with probabilities 
                # from corresponding distribution
                destination = np.random.choice(
                    range(tfs.number_zones),
                    p=distribution
                )
                
            # save the sampling outcome in transition matrix entry of
            # car-timestep combination
            tfs.transition_tensor[car, t, 1] = destination


def traveltime_and_distance_sampling(
    tfs,
    t
):

    """ Samples travel times and distances from normal Gaussian distributions.
    """

    # iterate over all cars
    for car in range(tfs.C):
        # get if car is sampled to drive
        drive = tfs.transition_tensor[car, t, 0]

        # sample driving duration and distance only if car drives
        if drive == 1:
            # get origin-destination sampling results
            origin = tfs.state_tensor[car, t]
            destination = int(
                round(
                    tfs.transition_tensor[car, t, 1]
                )
            )

            # if car drives within same city zone, use assumptions
            if origin == destination:
                # assumption: five minutes of travel time
                tfs.transition_tensor[car, t, 2] = 5 * 60
                # assumption: 1 km of travel distance
                tfs.transition_tensor[car, t, 3] = 1
            else:
                # Sampling travel times
                mean = tfs.datatensor_mean[origin, destination, t]
                if type(tfs.datatensor_stddev) != int:
                    std_deviation = tfs.datatensor_stddev[origin, destination, t]
                else:
                    std_deviation = 1 # CAUTION: assumption

                tfs.transition_tensor[car, t, 2] = abs(
                    np.random.normal(
                        mean,
                        std_deviation
                    )
                )
                
                # Sampling travel distances
                mean = tfs.od_distances.iloc[origin, destination]
                std_deviation = 0.1 # CAUTION: assumption
                tfs.transition_tensor[car, t, 3] = abs(
                    np.random.normal(
                        mean,
                        std_deviation
                    )
                )
                


