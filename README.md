NOTE ABOUT DATA
- Due to data protection reasons, actual dataset of emergencies cannot be released.
- The data provided here (sample_anon.csv and station_locations_latlng.csv) is a random sample of the original data. Furthermore, noise has been added on the time and location of all emergencies, and all names of stations, municipalities, and units and anonymized.
- The data provided is only meant to serve as a sample dummy data to demonstrate the functionalities of DAPI.

OPTIONS
- alias: the name to use for the generated files and results
- min_respones: minimum number of responses for a station to be considered
- outlier_threshold: minimum Mahalanobis distance for it to be considered an outlier
- restrict_same_municipality: set to True if the closer station has to be in the same municipality
- W, D: parameters for DAPI

RUNNING THE SOFTWARE
- The main.py file contains a sample usage of the tool.
- The tool assumes that OSRM is running at localhost for querying the road distances (https://phabi.ch/2020/05/06/run-osrm-in-docker-on-windows/).
- Whenever road distances are computed, results are stored in the generated/linear_model, generated/outlier_closer, and generated/outlier_response folders. If the road distances are already computed in the appropriate phase, they are not computed again. If you wish you compute the road distances again, you should delete the relevant file from those folders.
- Final list of potential inefficiencies are saved in the generated/results folder under the given alias.
