Bevpo stands for 'battery electric vehicle policy' and was first named this way 
by Anthony Patt and Marc Melliger. It is a package that provides city-scale car 
traffic and parking analyses. It utilizes a Hidden Markov Model to translate changes 
in travel time between different zones of a city throughout multiple time steps 
into probabilities of driving or not, as well as choosing a destination if driving. 
Given the joint distribution from these two multivariate probability distributions, 
bevpo then samples a vehicle fleet of arbitrary size. One key strength of bevpo 
is that it is able to model at arbitrary granularities in time and space, which 
are eventually given by the data provided to bevpo. 


### Installation:
```
pip install bevpo
```


### Docker:
For using bevpo within an Ubuntu docker container 
```
docker run -it aryandoustarsam/bevpo
```

For using bevpo with Jupyter notebook inside a docker container
```
docker run -it -p 3333:1111 -v ~/path_to_data/data:/data aryandoustarsam/bevpo:jupyter
[inside running container]: jupyter notebook --ip 0.0.0.0 --port 1111 --no-browser --allow-root
[in local machine browser]: localhost:3333 
[in local machine browser, type token shown in terminal]
```

### Usage guide:
At the core of bevpo stands the class bevpo.TrafficSystem. It bundles all 
properties and methods of the traffic system modeled with bevpo. Bellow is a list
of all parameters, attributes, methods and generated results.


<table>

  <tr>
    <th scope='row' colspan='2'> Parameters </th>
  </tr>
  
  <tr> 
    <td>
      <b>city_zone_coordinates (required)</b>: <br />  n_zones x 3 matrix
    </td>
    <td>
      A pandas.DataFrame containing the columns 'zone_id', 'zone_lat' and 
      'zone_long' indicating the respective geographic coordinates of each 
      city zone polygon centroid. 
    </td>
  </tr>
  
  <tr>
    <td>
      <b>od_mean_travel_time_list (required)</b>: <br /> list of n_data_t x 3 
      matrices
    <td>
      List of pandas.DataFrames containing the columns 'source_id', 'dest_id' 
      and 'mean_travel_time'. Each list element corresponds to measurements in
      one time stamp. List elements should appear in consecutive temporal order.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>od_stddev_travel_time_list (=None)</b>: <br />  list of 
      n_data_t x 3 matrices
    </td>
    <td>
      List of pandas.DataFrames containing the columns 'source_id', 'dest_id' and
      'stddev_travel_time'. Each list element corresponds to measurements in one
      time stamp. List elements should appear in consecutive temporal order.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>od_distances (=None)</b>: <br />  n_zones x n_zones
    </td>
    <td>
      Symmetric origin destination matrix with same zone IDs as columns and 
      indices in pandas.DataFrame
    </td>
  </tr>

  <tr>
    <td>
      <b>charging_profile (=None)</b>: <br />  list of floats > 0
    </td>
    <td>
      A charging profile for electric vehicles with one floating point entry per
      simulation time step. It hence has to have a length equal to
      len(od_mean_travel_time_list)
    </td>
  </tr>

  <tr>
    <td>
      <b>e_drive (=2)</b>: <br /> float > 0
    </td>
    <td>
      Exponential parameter describing the (exponential) functional relationship
      between changes in travel time in a particular city zone and the 
      probability that a car will drive.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>e_dest (=2)</b>: <br /> float > 0 
    </td>
    <td>
      Exponential parameter describing the (expoential) functional relationship
      between changes in travel time from one particular city zone to all other
      zones and the probabiity of choosing any of these other zones as a
      destination for a trip. 
    </td>
    
  </tr>
  
  <tr>
    <td>
      <b>p_min (=0.1)</b>: <br /> float in [0, p_max)
    </td>
    <td>
      The lower bound of probability distribution for p_drive. 
    </td>
  </tr>
  
  <tr>
    <td>
      <b>p_max (=0.9)</b>: <br /> float in (p_min, 1] 
    </td>
    <td>
      The upper bound of probability distribution for p_drive.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>cars_per_zone (=100)</b>: <br /> int > 0 
    </td>
    <td>
      The number of cars to be sampled for each city zone during traffic 
      simulation.
    </td>
  </tr>
  
</table>


<table>

  <tr>
    <th scope='row' colspan='2'> Attributes </th>
  </tr>
  
  <tr> 
    <td>
      <b>T</b>: <br /> int > 0
    </td>
    <td>
      Number of simulation time steps which results to len(od_mean_travel_time_list) 
    </td>
  </tr>
  
  <tr>
    <td>
      <b>number_zones</b>: <br /> int > 0 
    </td>
    <td>
      Number of city zones in od matrices which results to
      len(city_zone_coordinates)
    </td>
  </tr>
  
  <tr>
    <td>
      <b>datatensor_mean</b>: <br />  number_zones x number_zones x T 
    </td>
    <td>
      Sparse tensor that contains all provided OD values for mean travel time.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>datatensor_stddev</b>: <br /> number_zones x number_zones x T
    </td>
    <td>
       Sparse tensor that contains all provided OD values for standard deviation 
       of travel time.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>C</b>: <br /> int > 0  
    </td>
    <td>
      Size of vehicle fleet to be simulated, which results to number_zones *
      cars_per_zone  
    </td>
  </tr>
  
  <tr>
    <td>
      <b>state_tensor</b>: <br /> C x T 
    </td>
    <td>
      Tensor for saving states of traffic system which consists of the location
      of each simulated vehicle 
    </td>
  </tr>
  
  <tr>
    <td>
      <b>transition_tensor</b>: <br />  C x T x 4 
    </td>
    <td>
      Tensor for saving state transition of traffic system which consists of the
      driving, destination, travel time and travel distance of each simulated
      vehicle.
    </td>
  </tr>
  
</table>


<table>

  <tr>
    <th scope='row' colspan='2'> Methods </th>
  </tr>

  <tr>
    <td>
      <b>simulate_traffic()</b>:  
    </td>
    <td>
      Simulates the traffic system when called.
    </td>
  </tr>

  <tr> 
    <td>
      <b>calc_od_distances(city_zone_coordinates)</b>:   
    </td>
    <td>
      Called on initialization of class objects if od_distances=None. Calculates 
      the beeline distance between origin destination zones of a city if no 
      explicit matrix of such distances was past when initializing class object.
    </td>
  </tr>

  <tr> 
    <td>
      <b>create_datatensors()</b>:   
    </td>
    <td>
      Called by simulate_traffic() method. Transforms the list of od matrices 
      into a single datatensor. City zone indexing corresponds to respective 
      index positions in city_zone_coordinates.index. Time step indexing 
      corresponds to respective position of OD matrix in 
      od_mean_travel_time_list. If od_stddev_travel_time_list is available, the 
      same is done for this list too.
    </td>
  </tr>
  
  <tr> 
    <td>
      <b>save_tfs_results(path_to_results=None)</b>:   
    </td>
    <td>
      Saves the resulting plots and numeric values unter path_to_results when 
      called. If no path is passed, a directory called bevpo_results will be 
      created within current working directory, where results are stored.
    </td>
  </tr>
  
</table>


<table>

  <tr>
    <th scope='row' colspan='2'> Results </th>
  </tr>

  <tr>
    <td>
      <b>driving_map</b>: <br /> number_zones x T 
    </td>
    <td>
      Multivariate distribution of driving activity among all modeled city zones
      and sampled time steps. Distributions sum up to 1 across all time step and
      cityzones.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>parking_map</b>: <br /> number_zones x T 
    </td>
    <td>
      Multivariate distribution of parking cars among all modeled city zones
      and sampled time steps. Distributions sum up to 1 across all time step
      and city zones.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>charging_map</b>: <br /> number_zones x T 
    </td>
    <td>
      (Optional). Only created if parameter charging_profile is provided. 
      Multivariate distribution of charging cars among all modeled city zones
      and sampled time steps. Distribution sums up to 1 for all city zones and
      time steps.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>avg_driving_times</b>: <br /> array of length T+1 
    </td>
    <td>
       Average driving durations of all sampled trips. The first entry corresponds
       to the average over all time steps. The following entries correspond to
       the average over each simulated time step.
    </td>
  </tr>

  <tr>
    <td>
      <b>avg_driving_distances</b>: <br /> array of length T+1 
    </td>
    <td>
      Average driving distance of all sampled trips. The first entry corresponds
      to the average over all time steps. The following entries correspond to
      the average over each simulated time step.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>driving_share_lifetime</b>: <br />% 
    </td>
    <td>
      The average share of the lifetime of a car in which it drives.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>parking_share_lifetime</b>:<br /> % 
    </td>
    <td>
      The average share of the lifetime of a car in which it is parked.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>circadian_rhythm</b>: <br /> array of length T 
    </td>
    <td>
      The circadian rhythm, that is the percentual traffic activity, throughout
      all simulated time steps.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>distr_distances_per_t</b>: <br /> T x 10 
    </td>
    <td>
      A multivariate distribution of travelled distances for each time steps, 
      categorized into 10 bins.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>distr_durations_per_t</b>: <br /> T x 10 
    </td>
    <td>
      A multivariate distribution of travelled durations for each time steps, 
      categorized into 10 bins.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>distr_distances_total</b>: <br /> array of length 10 
    </td>
    <td>
      The total distribution of travelled distances categorized into 10 bins.
    </td>
  </tr>
 
  <tr>
    <td>
      <b>distr_durations_total</b>: <br /> array of length 10 
    </td>
    <td>
      The total distribution of travelled durations categorized into 10 bins.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>distr_bins_km</b>: <br /> array of length 10 
    </td>
    <td>
      The distance category of each bin in km unit.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>distr_bins_s</b>: <br /> array of length 10 
    </td>
    <td>
      The duration category of each bin in s unit.
    </td>
  </tr> 
</table>


### Examples:

Simulating traffic for exemplar Uber Movement travel time data, using only the 
minimum required information to pass to model.
```
import bevpo.datasets.prep_ubermovement as prep_data
import bevpo.trafficsystem as trafficsystem


### 1. Prepare Uber Data

# provide paths to Uber Movement data
path_to_rawdata = *insert path to Uber Movement .csv file here*
path_to_json_data = *insert path to Uber Movement .json file here*

# create city zone data from json file
city_zone_coordinates = (
    prep_data.create_city_zone_coordinates(path_to_json_data)
)

# create list of OD matrices from csv file
(
    od_mean_travel_time_list,
    od_std_travel_time_list
) = prep_data.create_od_matrix_lists(path_to_rawdata)


### 2. Simulate traffic 

# create object of class TrafficSystem and pass minimum required data
tfs = trafficsystem.TrafficSystem(
    city_zone_coordinates,
    od_mean_travel_time_list
)

# simulate traffic
tfs.simulate_traffic()

# save results
tfs.save_tfs_results()
```

Simulating traffic for from exemplar Uber Movement travel time data, using more 
than the minimum required information to pass to model, and customized parameters.
```
import bevpo.datasets.prep_ubermovement as prep_data
import bevpo.trafficsystem as trafficsystem


### 1. Prepare Uber Data

# provide paths to Uber Movement data
path_to_rawdata = *insert path to Uber Movement .csv file here*
path_to_json_data = *insert path to Uber Movement .json file here*
path_to_results = *insert path to desired results folder here*

# create city zone data from json file
city_zone_coordinates = (
    prep_data.create_city_zone_coordinates(path_to_json_data)
)

# create list of OD matrices from csv file
(
    od_mean_travel_time_list,
    od_std_travel_time_list
) = prep_data.create_od_matrix_lists(path_to_rawdata)

# create a charging profile of length 24
charging_profile = [
    5, 6, 7, 10, 10, 9, 8, 7, 6, 3, 2, 1, 5, 7, 4, 2, 1, 6, 4, 5, 5, 3, 7, 6
]

### 2. Simulate traffic 

# create object of class TrafficSystem and decleare custom parameters
tfs = trafficsystem.TrafficSystem(
    city_zone_coordinates,
    od_mean_travel_time_list,
    od_std_travel_time_list,
    charging_profile=charging_profile,
    e_drive=0.5,
    e_dest=2.3,
    p_min=0.085,
    p_max=0.81
)

# simulate traffic
tfs.simulate_traffic()

# save results
tfs.save_tfs_results(path_to_results)
```

