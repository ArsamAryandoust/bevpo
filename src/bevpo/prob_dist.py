import numpy as np

def calc_prob_dists(tfs):

    """ calculates the probabilities of driving p_drive and choosing a 
    destination p_dest.
    """


    create_distribution_p_drive(tfs)
    create_distribution_p_dest(tfs)
    #create_distribution_p_joint(tfs)


def create_distribution_p_drive(tfs):

    """ Calculates binomial probability distributions of driving for 
    each city zone and each time step. Note that city zones correspond
    to positions in datatensor and not the origional IDs from the
    city_zone_coordinates.index array.
    """

    p_drive = np.zeros(
        (
            tfs.number_zones,
            tfs.T
        )
    )
    mean_sum = np.zeros(
        (
            tfs.number_zones,
            tfs.T
        )
    )
    min_t = np.zeros(
        (
            tfs.number_zones
        )
    )
    max_t = np.zeros(
        (
            tfs.number_zones
        )
    )
    for source in range(tfs.number_zones):

        for time in range(tfs.T):
            mean_sum_term = 0
            counter = 0

            for dest in range(tfs.number_zones):
                if tfs.datatensor_mean[source, dest, time] > 0:
                    counter += 1
                    mean_sum_term = (
                        mean_sum_term
                        + tfs.datatensor_mean[source, dest, time] / (
                            tfs.od_distances.iloc[source, dest]
                        )
                    )

            if counter > 0:
                mean_sum[source, time] = mean_sum_term / counter

        value_array = mean_sum[source, :]
        min_t[source] = np.amin(value_array)
        max_t[source] = np.amax(value_array)

    for source in range(tfs.number_zones):
        if max_t[source] >0:

            for time in range(tfs.T):
                if mean_sum[source, time] > 0:
                    value = (
                        tfs.p_min
                        + (tfs.p_max - tfs.p_min) * (
                            (mean_sum[source, time] - min_t[source]) /
                            (max_t[source] - min_t[source])
                        )**tfs.e_drive
                    )
                else:
                    value = 0

                p_drive[source, time] = value

    tfs.p_drive = p_drive


def create_distribution_p_dest(tfs):

    """ Calculates probability distributions of choosing a destination.
    Note that city zones correspond to positions in datatensor and not 
    the origional IDs from the city_zone_coordinates.index array.
    """

    p_dest = np.zeros(
        (
            tfs.number_zones,
            tfs.number_zones,
            tfs.T
        )
    )
    min_x = np.zeros(
        (
            tfs.number_zones,
            tfs.number_zones
        )
    )
    max_x = np.zeros(
        (
            tfs.number_zones,
            tfs.number_zones
        )
    )
    normalization_factor = np.zeros(
        (
            tfs.number_zones,
            tfs.T
        )
    )
    for source in range(tfs.number_zones):

        for dest in range(tfs.number_zones):
            value_array = tfs.datatensor_mean[source, dest, :]
            max_x[source, dest] = np.amax(
                value_array
            )
            min_x[source, dest] = np.amin(
                value_array
            )

    for source in range(tfs.number_zones):

        for dest in range(tfs.number_zones):

            for time in range(tfs.T):
                if max_x[source, dest] > 0:
                    mean = tfs.datatensor_mean[source, dest, time]
                    p_dest[source, dest, time] = (
                        (
                            (mean - min_x[source, dest]) / (
                                max_x[source, dest] - min_x[source, dest]
                            )
                        )**tfs.e_dest
                    )

    for source in range(tfs.number_zones):

        for time in range(tfs.T):
            value_array = p_dest[source, :, time]
            normalization_factor[source, time] = np.sum(
                value_array
            )

    for source in range(tfs.number_zones):

        for time in range(tfs.T):
            if normalization_factor[source, time] > 0:
            
                for dest in range(tfs.number_zones):
                    p_dest[source, dest, time] = (
                        p_dest[source, dest, time] / (
                            normalization_factor[source, time]
                        )
                    )

    tfs.p_dest = p_dest


def create_distribution_p_joint(tfs):

    """ Calculates the joint probability distribution of driving and
    choosing a destination. Note that city zones correspond to positions 
    in datatensor and not the origional IDs from thecity_zone_coordinates.index 
    array.
    """
    p_joint = np.zeros(
        (
            tfs.number_zones,
            tfs.number_zones,
            tfs.T
        )
    )
    for source in range(tfs.number_zones):

        for time in range(tfs.T):
            multiplication_term = tfs.p_drive[source, time]
            if multiplication_term != 0:

                for dest in range(tfs.number_zones):
                    p_joint[source, dest, time] = (
                        tfs.p_dest[source, dest, time] * (
                            multiplication_term
                        )
                    )

    tfs.p_joint = p_joint
    
    
