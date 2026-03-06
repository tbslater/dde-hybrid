# Import required packages / files
from hybrid import HybridSim
import json
import os

def import_parameters():
	
	data_path = 'parameters'
	json_file = os.path.join(data_path, 'parameters.json')
	with open(json_file, "r", encoding="utf-8") as f:
		json_content = json.load(f)
	parameters = json_content

	return parameters

def main():
	
	# Load paarameters
	parameters = import_parameters()
	
	# Number of replications
	N_REPLICATIONS = 1
	
	# We'll store the interpolator objects
	results = {}
	results['stocks'] = []
	
	# Run model for N_REPLICATIONS
	for i in range(N_REPLICATIONS):
		parameters['main_seed'] = i
		model = HybridSim(parameters)
		model.simulate()
		results['stocks'].append(model.interpolator)

	return results

if __name__ == '__main__':
	main()