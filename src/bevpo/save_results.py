import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from mpl_toolkits import mplot3d

def save_results(tfs):

    """ Creates results folders, csv and plots and saves these in the respective
    folders.
    """
    
    create_resultsfolder(tfs)
    save_as_csv(tfs)
    save_as_plots(tfs)
    
    save_maps('parking', tfs, tfs.parking_map, 150000, 'parking_map.gif')
    save_maps('driving', tfs, tfs.driving_map, 150000, 'driving_map.gif')
    if tfs.charging_profile is not None:
        save_maps('charging', tfs, tfs.charging_map, 150000, 'charging_map.gif')


def create_resultsfolder(tfs):

    """ Creates a folder called bevpo_results on the tfs.path_to_results for
    saving both the plots and numeric values of the traffic simulation outcome.
    """

    if tfs.path_to_results.endswith('/'):
        tfs.path_to_bevpo_results = tfs.path_to_results
    else:
        tfs.path_to_bevpo_results = tfs.path_to_results + '/'

    tfs.path_to_csv = tfs.path_to_bevpo_results + 'numeric/'
    tfs.path_to_figures = tfs.path_to_bevpo_results + 'figures/'
    tfs.path_to_images = tfs.path_to_bevpo_results + 'images/'
    tfs.path_to_png = tfs.path_to_images + 'png/'
    tfs.path_to_pdf = tfs.path_to_images + 'pdf/'
    
    # create required folders for saving results if not existent yet
    if not os.path.isdir(tfs.path_to_bevpo_results):
        os.makedirs(tfs.path_to_bevpo_results)
    if not os.path.isdir(tfs.path_to_csv):
        os.mkdir(tfs.path_to_csv)
    if not os.path.isdir(tfs.path_to_figures):
        os.mkdir(tfs.path_to_figures)
    if not os.path.isdir(tfs.path_to_images):
        os.mkdir(tfs.path_to_images)
    if not os.path.isdir(tfs.path_to_png):
        os.mkdir(tfs.path_to_png)
    if not os.path.isdir(tfs.path_to_pdf):
        os.mkdir(tfs.path_to_pdf)
    
                                   
def save_as_csv(tfs):

    """ Saves the numeric results to csv files."""
    
    ### Save driving and  parking maps
    df_columns = []
    for t in range(tfs.T):
        column = 't={}'.format(t)
        df_columns.append(column)
        
    saving_path = tfs.path_to_csv + 'driving_map.csv'
    df = pd.DataFrame(
        tfs.driving_map, 
        index=tfs.city_zone_coordinates.index.values,
        columns=df_columns
    )
    df.to_csv(
        saving_path,
        index_label='zone_id'
    )
    
    saving_path = tfs.path_to_csv + 'parking_map.csv'
    df = pd.DataFrame(
        tfs.parking_map,
        index=tfs.city_zone_coordinates.index.values,
        columns=df_columns
    )
    df.to_csv(
        saving_path,
        index_label='zone_id'
    )
    
    ### Save charging maps for electric vehicles
    if tfs.charging_profile is not None:
        saving_path = tfs.path_to_csv + 'charging_map.csv'
        df = pd.DataFrame(
            tfs.charging_map,
            index=tfs.city_zone_coordinates.index.values,
            columns=df_columns
        )
        df.to_csv(
            saving_path,
            index_label='zone_id'
        )
    
    ### Save traffic system properties
    saving_path = tfs.path_to_csv + 'traffic_properties.csv'
    
    df_index = []
    df_index.append('total')
    for t in range(tfs.T):
        index = 't={}'.format(t)
        df_index.append(index)
        
    df_columns=[
        'avg driving time (min)',
        'avg driving distance (km)',
        'circadian rhythm (%)'
    ]
    
    avg_properties = np.column_stack(
        (
            tfs.avg_driving_times, 
            tfs.avg_driving_distances,
            np.append(
                '',
                np.round(tfs.circadian_rhythm * 100).astype(int)
            )
        )
    )
    df = pd.DataFrame(
        avg_properties, 
        columns=df_columns,
        index=df_index
    )
    df['driving share lifetime (%)'] = ''
    df['parking share lifetime (%)'] = ''
    
    df.iloc[0, 3] = tfs.driving_share_lifetime
    df.iloc[0, 4]  = tfs.parking_share_lifetime
    
    df.to_csv(saving_path)
    
    ### Save travel distributions
    saving_path = tfs.path_to_csv + 'distance_distributions.csv'
    df_index = []
    df_index.append('total (%)')
    for t in range(tfs.T):
        index = 't={}'.format(t)
        df_index.append(index)
    
    df_columns = []
    for bin_value in tfs.distr_bins_km:
        column = '{} km'.format(round(bin_value))
        df_columns.append(column)
    
    distr_distances = np.row_stack(
        (
            tfs.distr_distances_total, 
            tfs.distr_distances_per_t
        )
    )
    df = pd.DataFrame(
        distr_distances, 
        columns=df_columns,
        index=df_index
    )
    df.to_csv(saving_path)
    
    saving_path = tfs.path_to_csv + 'duration_distributions.csv'
    df_index = []
    df_index.append('total (%)')
    for t in range(tfs.T):
        index = 't={}'.format(t)
        df_index.append(index)
    
    df_columns = []
    for bin_value in tfs.distr_bins_s:
        column = '{} s'.format(round(bin_value))
        df_columns.append(column)
    
    distr_durations = np.row_stack(
        (
            tfs.distr_durations_total, 
            tfs.distr_durations_per_t
        )
    )
    df = pd.DataFrame(
        distr_durations, 
        columns=df_columns,
        index=df_index
    )
    df.to_csv(saving_path)


def save_maps(
    title, 
    tfs, 
    tfs_map, 
    scatter_factor, 
    gif_filename,
    fig_size=13,
    font=16
):

    """ Creates driving, parking and charging maps. """
    # create folders for saving files first
    tfs.path_to_png_map = tfs.path_to_png + title + '/'
    tfs.path_to_pdf_map = tfs.path_to_pdf + title + '/'
    
    if not os.path.isdir(tfs.path_to_png_map):
        os.mkdir(tfs.path_to_png_map)
    if not os.path.isdir(tfs.path_to_pdf_map):
        os.mkdir(tfs.path_to_pdf_map) 
    
    # save png and pdf images of the map first, and keep files in filename_list
    filename_list = []
    for t in range(tfs.T):

        plt.figure(
            figsize=(fig_size, fig_size)
        )

        plt.scatter(
            tfs.city_zone_coordinates['zone_long'].values,
            tfs.city_zone_coordinates['zone_lat'].values,
            tfs_map[:, t] * scatter_factor,
            alpha=0.7
        )
        plt.title(title + f'\n t={t}')
        plt.xlabel('longitude')
        plt.ylabel('latitude')

        filename_png = f't={t}.png'
        filename_pdf = f't={t}.pdf'
        filename_list.append(filename_png)
        saving_path_png = tfs.path_to_png_map + filename_png
        saving_path_pdf = tfs.path_to_pdf_map + filename_pdf
        plt.savefig(saving_path_png)
        plt.savefig(saving_path_pdf)
        plt.close()

    # create a gif from map images
    path_to_gif = tfs.path_to_figures + gif_filename

    with imageio.get_writer(path_to_gif, mode='I', fps=3) as writer:
        for filename in filename_list:
            loading_path = tfs.path_to_png_map + filename
            image = imageio.imread(loading_path)
            writer.append_data(image)


def save_as_plots(
    tfs,
    fig_size=13,
    font=16,
):

    """ Saves the resulting statistics to .png and .pdf plots."""  

    ### Save circadian rhythm
    
    saving_path_pdf = tfs.path_to_figures + 'circadian_rhythm.pdf'
    saving_path_png = tfs.path_to_figures + 'circadian_rhythm.png'
    fig = plt.figure(
        figsize=(fig_size, fig_size)
    )
    plt.plot(
        np.round(
            tfs.circadian_rhythm * 100
        ).astype(int)
    )
    plt.title(
        'Circadian rhythm of traffic',
        fontsize=font
    )
    plt.xlabel(
        'time',
        fontsize=font
    )
    plt.ylabel(
        'traffic activity [%]',
        fontsize=font
    )
    plt.xticks(
        fontsize=font
    )
    plt.yticks(
        fontsize=font
    )
    fig.savefig(saving_path_pdf)
    fig.savefig(saving_path_png)
    plt.close(fig)
    
    ### Save provided charging profile
    if tfs.charging_profile is not None:
        saving_path_pdf = tfs.path_to_figures + 'charging_profile.pdf'
        saving_path_png = tfs.path_to_figures + 'charging_profile.png'
        fig = plt.figure(
            figsize=(fig_size, fig_size)
        )
        plt.plot(
            np.round(
                tfs.charging_profile_dist * 100
            ).astype(int)
        )
        plt.title(
            'EV charging distribution over time',
            fontsize=font
        )
        plt.xlabel(
            'time',
            fontsize=font
        )
        plt.ylabel(
            'charging [%]',
            fontsize=font
        )
        plt.xticks(
            fontsize=font
        )
        plt.yticks(
            fontsize=font
        )
        fig.savefig(saving_path_pdf)
        fig.savefig(saving_path_png)
        plt.close(fig)    
    
    
    ### Save travel distance distributions
    
    saving_path_pdf = tfs.path_to_figures + 'travel_distance.pdf'
    saving_path_png = tfs.path_to_figures + 'travel_distance.png'
    fig = plt.figure(
        figsize=(fig_size, fig_size)
    )
    plt.bar(
        range(len(tfs.distr_bins_km)),
        tfs.distr_distances_total
    )
    plt.xticks(
        range(len(tfs.distr_bins_km)),
        np.round(tfs.distr_bins_km).astype(int),
        fontsize=font
    )
    
    plt.title(
        'Travelled distances per trip',
        fontsize=font
    )
    plt.xlabel(
        'distance [km]',
        fontsize=font
    )
    plt.yticks(
        fontsize=font
    )
    plt.ylabel(
        'trips [%]',
        fontsize=font
    )
    fig.savefig(saving_path_pdf)
    fig.savefig(saving_path_png)
    plt.close(fig)
    
    
    ### Save travel duration distributions
    
    saving_path_pdf = tfs.path_to_figures + 'travel_duration.pdf'
    saving_path_png = tfs.path_to_figures + 'travel_duration.png'
    fig = plt.figure(
        figsize=(fig_size, fig_size)
    )
    plt.bar(
        range(len(tfs.distr_bins_s)),
        tfs.distr_durations_total
    )
    plt.xticks(
        range(len(tfs.distr_bins_s)),
        np.round(tfs.distr_bins_s/60, 2),
        fontsize=font
    )
    plt.yticks(
        fontsize=font
    )
    plt.title(
        'Travelled durations per trip',
         fontsize=font
    )
    plt.xlabel(
        'duration [min]',
        fontsize=font
    )
    plt.ylabel(
        'trips [%]',
        fontsize=font
    )
    fig.savefig(saving_path_pdf)
    fig.savefig(saving_path_png)
    plt.close(fig)
    
    
    saving_path_pdf = tfs.path_to_figures + 'travel_distance_per_t.pdf'
    saving_path_png = tfs.path_to_figures + 'travel_distance_per_t.png'
    fig = plt.figure(
        figsize=(fig_size, fig_size)
    )
    ax = plt.axes(projection='3d')
    ax.plot_surface(
        tfs.distr_bins_km,
        np.expand_dims(np.arange(tfs.T), 1),
        tfs.distr_distances_per_t,
    )
    ax.set_title(
        'Travel distance distribution',
        fontsize=font
    )
    ax.set_xlabel(
        'travel distance [km]',
        fontsize=font
    )
    ax.set_ylabel(
        'time step (t)',
        fontsize=font
    )
    ax.set_zlabel(
        'sampled cars',
        fontsize=font
    )
    fig.savefig(saving_path_png)
    fig.savefig(saving_path_pdf)
    plt.close(fig)
    
    saving_path_pdf = tfs.path_to_figures + 'travel_duration_per_t.pdf'
    saving_path_png = tfs.path_to_figures + 'travel_duration_per_t.png'
    fig = plt.figure(
        figsize=(fig_size, fig_size)
    )
    ax = plt.axes(projection='3d')
    ax.plot_surface(
        tfs.distr_bins_s,
        np.expand_dims(np.arange(tfs.T), 1),
        tfs.distr_durations_per_t,
    )
    ax.set_title(
        'Travel duration distribution',
        fontsize=font
    )
    ax.set_xlabel(
        'travel time [s]',
        fontsize=font
    )
    ax.set_ylabel(
        'time step (t)',
        fontsize=font
    )
    ax.set_zlabel(
        'sampled cars',
        fontsize=font
    )
    fig.savefig(saving_path_png)
    fig.savefig(saving_path_pdf)
    plt.close(fig)
        
