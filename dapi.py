import tqdm as tqdm
import pandas as pd
import numpy as np
from os.path import exists
from pyproj import Transformer
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import requests 
import json
from folium.plugins import BeautifyIcon

data = None
options = {
	'alias': 'df',
	'min_responses': 100,
	'outlier_threshold': 1.5,
	'restrict_same_municipality': False,
	'W': 2, # in number of hours
	'D': 7 # in number of days
}

generated = {
	'df_year': None,
	'df_outliers': None,
	'rpv_table': None,
	'busy_table': None,
	'busy_scores_in_context': None,
	'closer_mask': None,
	'means': None,
	'station_municipalities': None,
	'station_locations': None
}

url = 'http://localhost:5000'
def get_road_distance(source, dest):
	slat, slong = source[0], source[1]
	dlat, dlong = dest[0], dest[1]
	parameter_string = '{0},{1};{2},{3}'.format(slong, slat, dlong, dlat)
	r = requests.get(url + '/route/v1/driving/{0}?overview=false'.format(parameter_string))
	#print(url + '/route/v1/driving/{0}?overview=false'.format(parameter_string))
	distance = -1
	try:
			distance = json.loads(r.text)['routes'][0]['distance']
	except:
			print(f'error: {source} to {dest}')
	return distance

def initialize(df, **kwargs):
	global data
	data = df
	for arg in kwargs:
		options[arg] = kwargs[arg]

def log(message, verbose=True):
	if verbose:
		print(message, flush=True)

def haversine_np(src, dest):
	lat1 = [i[0] for i in src]
	lon1 = [i[1] for i in src]
	lat2 = [i[0] for i in dest]
	lon2 = [i[1] for i in dest]
	lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
	c = 2 * np.arcsin(np.sqrt(a))
	km = 6367 * c
	return km

def get_response_count(df_year, freq):
	response_count_table = df_year.set_index('time').groupby(['station', pd.Grouper(freq='1H')]).agg(count=('time', 'count')).reset_index()\
		.pivot(index='time', columns='station', values='count').fillna(0).astype('int16')\
		.rolling(min_periods=1, window=freq).sum()
	return response_count_table

def normalizer_all(rpv_table):
    res_mean = np.tile(rpv_table.mean().to_numpy().reshape(len(rpv_table.columns), 1), len(rpv_table)).transpose()
    res_std = np.tile(rpv_table.std(ddof=1).to_numpy().reshape(len(rpv_table.columns), 1), len(rpv_table)).transpose()
    res_mean = pd.DataFrame(data=res_mean, index=rpv_table.index, columns=rpv_table.columns)
    res_std = pd.DataFrame(data=res_std, index=rpv_table.index, columns=rpv_table.columns)
    return res_mean, res_std

def normalizer_same_time(rpv_table):
    hours = pd.unique(rpv_table.index.hour)
    res_mean = np.zeros((len(rpv_table), len(rpv_table.columns)))
    res_std = np.zeros((len(rpv_table), len(rpv_table.columns)))
    for h in hours:
        s = rpv_table.iloc[rpv_table.index.hour == h]
        smean = s.mean().to_numpy()
        sstd = s.std(ddof=1).to_numpy()
        res_mean[rpv_table.index.hour == h] = smean
        res_std[rpv_table.index.hour == h] = sstd
    res_mean = pd.DataFrame(data=res_mean, index=rpv_table.index, columns=rpv_table.columns)
    res_std = pd.DataFrame(data=res_std, index=rpv_table.index, columns=rpv_table.columns)
    return res_mean, res_std

def get_rpv_distribution(df_year, freq, D):
	vehicle_count = df_year.set_index('time').groupby([pd.Grouper(freq='1D'), 'station'])\
		.agg(count=('unit', lambda x: len(pd.unique(x)))).reset_index()\
		.pivot(index='station', columns='time', values='count').fillna(0).astype('int16')\
		.rolling(min_periods=1, window=D*2, center=True).max()\
		.copy(deep=True)
	rpv_table = df_year.set_index('time').groupby(['station', pd.Grouper(freq='5min')]).agg(count=('station', 'count')).reset_index()\
		.pivot(index='time', columns='station', values='count').fillna(0).astype('int16')\
		.rolling(min_periods=1, window=freq).sum().copy(deep=True)
	for station in rpv_table.columns:
		divider = pd.to_datetime(rpv_table[station].index.date).map(vehicle_count.loc[station])
		rpv_table[station] = rpv_table[station] / divider
	rpv_table.replace([np.inf, -np.inf], np.nan, inplace=True)
	rpv_table = np.round(rpv_table, 2)
	return rpv_table, vehicle_count

def get_busy_scores(rpv_table, normalizer=normalizer_all):
    rpv_means, rpv_stds = normalizer(rpv_table)
    busy_table = rpv_table.copy(deep=True)
    busy_table = ((busy_table - rpv_means) / rpv_stds)
    busy_table.replace([np.inf, -np.inf], np.nan, inplace=True)
    return busy_table

def execute():
	alias = options['alias']
	threshold = options['outlier_threshold']
	time_window = options['W']
	time_window = f'{time_window}H'
	D = options['D']

	log(f'Copying dataset...')
	df = data.copy(deep=True)
	
	log(f'Loading station locations...')
	station_locations = pd.read_csv('station_locations_latlng_anon.csv')
	def get_station_location(station_name):
		return tuple(station_locations.set_index('station').loc[station_name][['latitude', 'longitude']])
	
	station_locations_indexed = station_locations.set_index('station')[['latitude', 'longitude']]
	locations = station_locations.set_index('station')[['latitude', 'longitude']]\
    .apply(lambda x: [[x['latitude']], [x['longitude']]], axis=1)
	mapper = station_locations.groupby('station').apply(lambda x: (x['latitude'].iloc[0], x['longitude'].iloc[0]))
	df['station_location'] = df['station'].map(mapper)

	log(f'Dropping instances with no station location information...')
	previous_length = len(df)
	missing = np.unique(df[df['station_location'].isna()]['station'])
	log(f'Missing station locations for: {missing}')
	df = df.dropna(subset=['station_location']).copy(deep=True)
	current_length = len(df)
	log(f'{previous_length - current_length} row/s dropped!')
	
	log(f'Computing Haversine distance to responding station...')
	df['location'] = list(zip(df['latitude'], df['longitude']))
	df['haversine_distance'] = haversine_np(df['station_location'], df['location'])
	
	source = f'./generated/linear_model/{alias}.csv'
	if exists(source):
		log('File found! Loading sample for linear model...')
	else:
		log(f'Randomly sampling 10000 instances to determine estimation bounds...')
		df_sample = df.sample(10000)
		road_distances = []
		missings = []
		with tqdm.tqdm(total=len(df_sample)	) as pbar:
			for target_index in range(len(df_sample)):
				try:
					st = df_sample.iloc[target_index]['station']
					s = tuple(station_locations_indexed.loc[st])
					d = tuple(df_sample.iloc[target_index][['latitude', 'longitude']])
					res = get_road_distance(s, d)
					road_distances.append(res)
				except:
					road_distances.append(-1)
					if st not in missings:
						log(f'Warning: missing key for station {st}')
						missings.append(st)
				pbar.update(1)
		results = pd.DataFrame({'haversine_distance': df_sample['haversine_distance'].tolist(), 'road_distance': np.array(road_distances)/1000.0})
		results.to_csv(source)

	log(f'Determining linear model for predicting road distance...')
	df_sample = pd.read_csv(source, index_col=0)
	X = df_sample['haversine_distance'].to_numpy().reshape(-1,1)
	y = df_sample['road_distance'].to_numpy()
	linear_model = LinearRegression().fit(X,y)
	z = linear_model.predict(X)
	linear_model_residue_mean = np.mean(z-y)
	linear_model_residue_std = np.std(z-y, ddof=1)
	linear_model = LinearRegression().fit(X,y)
	residue_lower_bound = norm.ppf(.005,loc=linear_model_residue_mean, scale=linear_model_residue_std)
	residue_upper_bound = norm.ppf(.995,loc=linear_model_residue_mean, scale=linear_model_residue_std)
	log(f'Residue bounds: ({residue_lower_bound}, {residue_upper_bound})')
	log(f'Error bound: {residue_lower_bound}')

	log(f'Dropping unavailable station locations and/or road distances...')
	previous_length = len(df)
	df = df.dropna(subset=['station_location'])
	current_length = len(df)
	log(f'{previous_length - current_length} row/s dropped!')
		
	log(f'Filtering minimum responses...')
	previous_length = len(df)
	#df = df.query('road_distance >= 0.0')
	x = df.groupby('station').agg({'time': 'count'})
	to_remove = x[(x < options['min_responses'])['time']].index
	df = df.query('station not in @to_remove')
	current_length = len(df)
	log(f'{previous_length - current_length} row/s dropped!')

	log(f'Computing Mahalanobis distance to responding station...')
	response_locations = df.groupby('station').agg(latitude=('latitude', list), longitude=('longitude', list))
	response_locations = response_locations[response_locations['latitude'].apply(lambda x: len(x) >= options['min_responses'])]
	cov_matrices = response_locations.apply(\
		lambda x: np.cov(np.array(x['latitude'], dtype='float16'), np.array(x['longitude'], dtype='float16')), axis=1)
	means = response_locations.apply(\
		lambda x: np.array([np.mean(x['latitude']), np.mean(x['longitude'])]).reshape(2,1), axis=1)
	df_target = df.query('station in @response_locations.index').copy(deep=True)
	m = np.concatenate(means[df_target['station'].tolist()].to_numpy()).reshape(len(df_target),2)
	a = df_target[['latitude', 'longitude']].to_numpy() - m
	cov_matrices_inverse = pd.Series()
	
	for c in cov_matrices.index:
		inv = np.linalg.pinv(cov_matrices[c])
		cov_matrices_inverse.loc[c] = inv
	cov_matrices_inverse[df_target.iloc[0]['station']]
	b = np.concatenate(cov_matrices_inverse[df_target['station'].tolist()].to_numpy())
	left = b[[i for i in range(0,len(df_target)*2,2)],:].transpose()
	right = b[[i for i in range(1,len(df_target)*2,2)],:].transpose()
	left_res = (a * left.transpose()).sum(-1).reshape(-1,1)
	right_res = (a * right.transpose()).sum(-1).reshape(-1,1)
	d = np.concatenate([left_res, right_res], axis=1)
	df_target['distance'] = np.sqrt((d * a).sum(1)).astype('float16')

	log(f'Detecting potential outliers using threshold {threshold}...')
	df_outliers = df_target.query('distance > @threshold').copy(deep=True)
	log(f'{len(df_outliers)} outliers detected.')
	
	log(f'Computing Haversine distance from every station...')
	haversine_distances = []
	with tqdm.tqdm(total=len(means.index)) as pbar:
		for s in means.index:
			sloc = [get_station_location(s)]
			dist = haversine_np(df_outliers['location'], sloc)
			haversine_distances.extend(dist)
			pbar.update(1)
	haversine_distances = np.array(haversine_distances).reshape(len(means.index), len(df_outliers))
	
	log(f'Filtering list based on predicted road distance...')
	station_municipalities = df.groupby('station').apply(lambda x: x['municipality'].value_counts().index[0])
	if options['restrict_same_municipality']:
		a = np.tile(station_municipalities.to_numpy().reshape(len(station_municipalities),1),len(df_outliers))
		b = np.tile(df_outliers['municipality'].to_numpy().reshape(len(df_outliers), 1), len(station_municipalities)).transpose()
		haversine_distances_adjusted_municipality = haversine_distances + ((a!=b) * 99999)
	else:
		haversine_distances_adjusted_municipality = haversine_distances
	df_outliers['min_haversine_distance'] = np.min(haversine_distances_adjusted_municipality, axis=0)
	df_outliers['min_haversine_distance_station'] =\
		station_municipalities.index[np.argmin(haversine_distances_adjusted_municipality, axis=0)].to_numpy()
	# df_outliers = df_outliers.query('station != min_haversine_distance_station').copy(deep=True)

	log(f'Dropping outliers based on prediction')	
	X = df_outliers['haversine_distance'].to_numpy().reshape(-1,1)
	pred = np.array(linear_model.predict(X))
	X = df_outliers['min_haversine_distance'].to_numpy().reshape(-1,1)
	min_pred = np.array(linear_model.predict(X))
	print('pred')
	print(pred)
	print('min_pred')
	print(min_pred)
	to_check = min_pred + residue_lower_bound < pred - residue_lower_bound
	previous_length = len(df_outliers)
	df_outliers = df_outliers.iloc[to_check].copy(deep=True)
	current_length = len(df_outliers)
	log(f'{previous_length - current_length} row/s dropped from {previous_length}!')

	log(f'Checking if road distance data is available for outliers...')
	source = f'./generated/outlier_response/{alias}.csv'
	if exists(source):
		log('File found! Loading road distances...')
	else:
		log('File not found! Extracting road distances...')
		road_distances = []
		with tqdm.tqdm(total=len(df_outliers)) as pbar:
			for target_index in range(len(df_outliers)):
				s = tuple(station_locations_indexed.loc[df_outliers.iloc[target_index]['station']])
				d = tuple(df_outliers.iloc[target_index][['latitude', 'longitude']])
				res = get_road_distance(s, d)
				road_distances.append(res)
				pbar.update(1)
		res = pd.DataFrame({'road_distance': road_distances}, index=df_outliers.index)
		res.to_csv(source)

	df_outliers_road_distances = pd.read_csv(source, index_col=0)
	df_outliers['road_distance'] = df_outliers_road_distances['road_distance'] / 1000.0

	previous_length = len(df_outliers)
	df_outliers = df_outliers.query('road_distance >= 0').dropna(subset=['road_distance']).copy(deep=True)
	current_length = len(df_outliers)
	log(f'{previous_length - current_length} row/s dropped!')

	log(f'Predicting the road distance between outliers and every station...')
	predicted_road_distances = []
	with tqdm.tqdm(total=len(means.index)) as pbar:
		for s in means.index:
			sloc = [get_station_location(s)]
			dist = haversine_np(df_outliers['location'], sloc)
			predicted_road_distances.extend(dist)
			pbar.update(1)
	
	z = linear_model.predict(np.array(predicted_road_distances).reshape(-1,1))
	predicted_road_distances = z.reshape(len(means.index), len(df_outliers)) - residue_upper_bound
	response_road_distances = np.tile(df_outliers['road_distance'].to_numpy()\
		.reshape(len(df_outliers), 1), len(means.index)).transpose()
	predicted_distance_mask = predicted_road_distances <\
		response_road_distances #+ np.abs(residue_lower_bound - linear_model_residue_mean)
	response_municipalities =\
		np.tile(df_outliers['municipality'].to_numpy().reshape(len(df_outliers), 1), len(means.index)).transpose()
	ref_municipalities =\
		np.tile(station_municipalities[means.index].to_numpy().reshape(len(means.index), 1), len(df_outliers))
	same_municipality_mask = response_municipalities == ref_municipalities
	response_stations =\
		np.tile(df_outliers['station'].to_numpy().reshape(len(df_outliers), 1), len(means.index)).transpose()
	ref_stations =\
		np.tile(means.index.to_numpy().reshape(len(means.index), 1), len(df_outliers))
	different_station_mask = response_stations != ref_stations
	if options['restrict_same_municipality']:
		mask = predicted_distance_mask & same_municipality_mask
	else:
		mask = predicted_distance_mask
	mask = pd.DataFrame(mask, index=means.index, columns=[i for i in range(len(df_outliers))])

	log(f'Checking if road distance data is available for outliers and every station...')
	source = f'./generated/outlier_closer/{alias}.csv'
	if exists(source):
		log('File found! Loading road distances...')
	else:
		log('File not found! Extracting road distances...')
		results = np.ones((len(mask), len(df_outliers))) * 99999000
		a = mask.to_numpy()
		with tqdm.tqdm(total=len(df_outliers)) as pbar:
			for i in range(len(df_outliers)):
				current_mask = a[:,i]
				indices = list(np.where(current_mask)[0])
				stations = list(mask.index[current_mask])
				d = tuple(df_outliers[['latitude', 'longitude']].iloc[i])
				for j, idx in zip(stations, indices):
					s = tuple(station_locations_indexed.loc[j])
					dist = get_road_distance(s, d)
					results[idx][i] = dist
				pbar.update(1)
		pd.DataFrame(data=results, index=mask.index, columns=mask.columns).to_csv(source)
	
	log(f'Computing relevant stations...')
	actual_road_distances = pd.read_csv(source, index_col=0).to_numpy() / 1000.0
	closer_mask = (actual_road_distances < response_road_distances) & different_station_mask
	relevant_list = []
	busy_score_list = []
	distances_list = []
	with tqdm.tqdm(total=len(df_outliers)) as pbar:
		for i in range(len(df_outliers)):
			relevant_list.append(np.array(means.index)[closer_mask[:,i]])
			pbar.update(1)
	df_outliers['relevant'] = relevant_list

	log(f'Computing RPV and busy scores per station with time frame: {time_window}')
	rpv_table, vehicle_count = get_rpv_distribution(df, time_window, D)
	busy_table = get_busy_scores(rpv_table, normalizer=normalizer_same_time)

	log(f'Computing busy scores in context...')
	stations_ref = np.tile(np.array(busy_table.columns),(len(df_outliers),1)).transpose()
	outlier_times = np.tile(np.array(df_outliers['time'].dt.floor('5min')),(len(busy_table.columns),1))
	mapper = { a:b for a,b in zip(busy_table.index.to_numpy(),[i for i in range(len(busy_table))]) }
	u,inv = np.unique(outlier_times,return_inverse = True)
	a = np.array([mapper[x] for x in u])[inv].reshape(outlier_times.shape)
	mapper = { a:b for a,b in zip(busy_table.columns,[i for i in range(len(busy_table.columns))]) }
	u,inv = np.unique(stations_ref,return_inverse = True)
	b = np.array([mapper[x] for x in u])[inv].reshape(stations_ref.shape)
	busy_scores = busy_table.to_numpy()[a.flatten(), b.flatten()].reshape(outlier_times.shape)
	
	log(f'Computing busy scores and distances for relevant stations...')
	with tqdm.tqdm(total=len(df_outliers)) as pbar:
		for i in range(len(df_outliers)):
			busy_score_list.append(busy_scores[:,i][closer_mask[:,i]])
			distances_list.append(actual_road_distances[:,i][closer_mask[:,i]])
			pbar.update(1)
	df_outliers['relevant_busy_score'] = busy_score_list
	df_outliers['relevant_distances'] = distances_list

	log(f'Dropping outliers with no closer stations...')
	selected_indices = []
	i = 0
	for idx, row in df_outliers.iterrows():
		if len(row['relevant']) > 0:
			selected_indices.append(i)
		i = i + 1
	previous_length = len(df_outliers)
	df_outliers = df_outliers.iloc[selected_indices].copy(deep=True)
	current_length = len(df_outliers)
	log(f'{previous_length - current_length} row/s dropped!')
	
	results_path = f'./generated/results/{alias}'
	df_outliers.to_pickle(f'{results_path}_potential_inefficiencies.pkl')
	df_outliers.to_csv(f'{results_path}_potential_inefficiencies.csv')
	#rpv_table.to_pickle(f'{results_path}_rpv_table.pkl')
	#busy_table.to_pickle(f'{results_path}_busy_table.pkl')
	#vehicle_count.to_pickle(f'{results_path}_vehicle_count.pkl')
	
	generated['rpv_table'] = rpv_table
	generated['busy_table'] = busy_table
	generated['vehicle_count'] = vehicle_count
	generated['busy_scores_in_context'] = busy_scores
	generated['df_year'] = df
	generated['df_outliers'] = df_outliers
	generated['closer_mask'] = closer_mask
	generated['means'] = means
	generated['station_municipalities'] = station_municipalities
	generated['station_locations'] = station_locations_indexed

	print(generated['df_outliers'].head())

	log(f'Done! Results saved in generated/results')
	
def display_state():
	if data is not None:
		print(f'Data: {len(data)} rows')
	else:
		print('No data loaded.')
	print(options)