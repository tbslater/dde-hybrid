# Import required packages / files
from hybrid.hybrid import HybridSim
from sd.model import SDModel
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def import_parameters():
	
	data_path = '../parameters'
	json_file = os.path.join(data_path, 'parameters.json')
	with open(json_file, "r", encoding="utf-8") as f:
		json_content = json.load(f)
	parameters = json_content

	return parameters

def main():
	
	start = time.time()
	
	# Load paarameters
	parameters = import_parameters()

	# Time domain
	time_domain = np.linspace(0, parameters['general']['horizon'], 1001)

	# System dynamics model only
	print('Running system dynamics model...')

	# Run the models
	sd_pars = parameters['system_dynamics']
	results = {}
	model = SDModel(sd_pars, method='interp')
	model.solve(100)
	results['interpolation'] = model.interpolator(time_domain)[2]
	for i in range(1,5):
	    sd_pars['delay_order'] = i
	    model = SDModel(sd_pars, method='chain')
	    model.solve(100)
	    results[f'Order_{i}'] = model.interpolator(time_domain)[2]

	# Plot the results
	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot()
	ax.plot(time_domain, results['interpolation'], label='Interpolation', 
	        lw=2, c='black', linestyle='-')
	colours = ['C0', 'C1', 'C2', 'C4']
	styles = ['-', ':', '--', '-.']
	for i in range(1,5):
	    ax.plot(time_domain, results[f'Order_{i}'], label = f'Erlang: n={i}', 
	            alpha=0.75, linestyle=styles[i-1], lw=2, c='black')
	ax.set_xlabel('Days')
	ax.set_ylabel('Number of individuals')
	ax.legend(loc=[0.025,1.05], ncol=5)
	ax.grid(linestyle=':')
	fig.savefig('../figures/pipeline-delay-plt.png')

	print('Completed.')

	# Now run the hybrid model
	print('Running hybrid simulation model...')
	
	# Number of replications
	N_REPLICATIONS = 10

	# Scenarios
	scenarios = ['Baseline', 'Increased Quarantine', 
				 'Increased Quarantine + Vaccinations']
	results = np.zeros(len(scenarios))

	for i, scenario in enumerate(scenarios):

		print(f'{scenario} Scenario.')

		# Change relevant parameters
		if i==1 or i==2: 
			parameters['system_dynamics']['quarantine_fraction'] = 0.9
		if i==2:
			parameters['agent_based']['max_daily_vax'] = 1000

		max_infections = np.zeros(N_REPLICATIONS)

		# Store maximum no. of infections for each replication
		for j in range(N_REPLICATIONS):
			parameters['general']['main_seed'] = j
			model = HybridSim(parameters)
			model.simulate()
			infections = model.interpolator(time_domain)[1]
			max_infections[j] = np.max(infections)
			print(f'Replication {j+1} completed.')

		results[i] = np.mean(max_infections)

	plot_df = pd.DataFrame()
	plot_df['scenario'] = scenarios
	plot_df['peak_infections'] = results

	# Bar chart for peak infections
	fig = plt.figure(figsize=(10,4))
	ax = fig.add_subplot()
	ax.bar(plot_df['scenario'], plot_df['peak_infections'], 
		   color='darkgrey', edgecolor='black')
	ax.set_ylabel('Peak Number of Infections')
	fig.savefig('../figures/peak-infections-plt.png')

	print('Completed.')

	end = time.time()
	elapsed = end - start
	print(f'Total run time: {np.round(elapsed/60, decimals=1)} minutes.')

if __name__ == '__main__':
	main()