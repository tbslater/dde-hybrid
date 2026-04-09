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
from joblib import Parallel, delayed, cpu_count

def import_parameters():
	
	data_path = '../parameters'
	json_file = os.path.join(data_path, 'parameters.json')
	with open(json_file, "r", encoding="utf-8") as f:
		json_content = json.load(f)
	parameters = json_content

	return parameters

def run_sd_model(pars, method='interp', delay_order=None):

    pars['delay_order'] = delay_order
    model = SDModel(pars, method=method)
    start = time.time()
    model.solve(80)
    end = time.time()
    elapsed = end - start
    return elapsed

def run_hybrid_model(pars, time_domain, seed):
    pars['general']['main_seed'] = seed
    model = HybridSim(pars)
    model.simulate()
    infections = model.interpolator(time_domain)[1]
    return np.max(infections)

def main():
	
	main_start = time.time()
	
	# Load parameters
	parameters = import_parameters()

	# Time domain
	time_domain = np.linspace(0, parameters['general']['horizon'], 1001)

	# System dynamics model only
	print('Running system dynamics model...')

	print('Plotting Erlang distributions up to order 100.')

	# Run the SD models for different methods of solving the pipeline delay
	results = {}
	model = SDModel(parameters['system_dynamics'], method='interp')
	model.solve(parameters['general']['horizon'])
	results['interpolation'] = model.interpolator(time_domain)[2]
	print(f'Method: interpolation. Min. value: {min(results['interpolation'])}.')
	values = [1,2,3,4,5,10,25,50,100]
	for i in values:
	    parameters['system_dynamics']['delay_order'] = i
	    model = SDModel(parameters['system_dynamics'], method='LCT')
	    model.solve(parameters['general']['horizon'])
	    results[f'Order_{i}'] = model.interpolator(time_domain)[2]
	    minimum = min(results[f'Order_{i}'])

	# Plot the results
	fig = plt.figure(figsize=(12,5))
	ax = fig.add_subplot()
	ax.plot(time_domain, results['interpolation'], label='Interpolation', 
	        lw=3, c='black', linestyle='-')
	for i in values:
	    ax.plot(time_domain, results[f'Order_{i}'], label = f'Erlang: n={i}', 
	            alpha=0.6, linestyle='-.', lw=2.5)
	ax.set_xlabel('Days', fontsize=12)
	ax.set_ylabel('Number of individuals', fontsize=12)
	plt.xticks(fontsize=11)
	plt.yticks(fontsize=11)
	ax.legend(loc=[0,1.025], ncol=5, fontsize=12)
	ax.grid(linestyle=':')
	data_path = '../figures/pipeline-delay-plt.png'
	fig.savefig(data_path, bbox_inches='tight', dpi=300)

	print('Completed.')

	print('Testing run times and errors for the SD model.')

	# Calculate run time across 100 repeats for n=1,10,100,1000
	N_RUNS = 100
	values = [10**x for x in range(4)]
	results = np.zeros((len(values)+1)*N_RUNS).reshape((len(values)+1, N_RUNS))
	for run in range(N_RUNS):
		results[0, run] = run_sd_model(parameters['system_dynamics'])
		for i, j in enumerate(values):
			results[i+1, run] = run_sd_model(parameters['system_dynamics'], 
											 method='LCT', delay_order=j)
		if (run+1) % 10 == 0:
			print(f'{((run+1)/N_RUNS) * 100}% complete.')

	# Calculate means and 90% confidence intervals
	means = np.mean(results, axis=1)
	means = np.round(means, decimals=4)
	sorted_results = np.sort(results)
	lowers = np.round(sorted_results[:,9], decimals=4)
	uppers = np.round(sorted_results[:,89], decimals=4)

		
	# Store results in a table
	comp_results = pd.DataFrame()
	comp_results['Method'] = ['Interpolation', 'Erlang: n=1', 'Erlang: n=10',
							  'Erlang: n=100', 'Erlang: n=1000']
	comp_results['Mean'] = means
	comp_results['Lower'] = lowers
	comp_results['Upper'] = uppers

	# Calculate errors for Erlang approximations
	q_vals = np.zeros((len(values)+1)*len(time_domain))
	q_vals = q_vals.reshape(len(values)+1, len(time_domain))
	model = SDModel(parameters['system_dynamics'], method='interp')
	model.solve(80)
	q_vals[0] = model.interpolator(time_domain)[2]
	for i, j in enumerate(values):
	    parameters['system_dynamics']['delay_order'] = j
	    model = SDModel(parameters['system_dynamics'], method='LCT')
	    model.solve(80)
	    q_vals[i+1] = model.interpolator(time_domain)[2]
	max_error = np.round(np.max(abs(q_vals[0] - q_vals[1:]), axis=1), decimals=2)
	max_error = np.concatenate(([None], max_error))
	comp_results['Error'] = max_error

	# Save pandas data frame as CSV
	data_path = '../figures/comp-results.csv'
	comp_results.to_csv(data_path, index=False)

	# Now run the hybrid model
	print('Running hybrid simulation model...')

	# Number of replications
	N_REPLICATIONS = 10
	
	# Scenarios
	methods = ['interp', 'LCT']
	scenarios = ['Baseline', 'Increased Quarantine', 
	             'Increased Quarantine + Vaccinations']
	results_dict = {}
	
	for method in methods:
	
	    parameters['system_dynamics']['method'] = method
	    print(f'Method: {method}.')
	    if method == 'LCT':
	        parameters['system_dynamics']['delay_order'] = 100
	
	    results = np.zeros(len(scenarios))
	
	    for i, scenario in enumerate(scenarios):
	    
	        print(f'Currently running: {scenario} Scenario.')
	    
	        # Change relevant parameters
	        if i == 0:
	            parameters['system_dynamics']['quarantine_fraction'] = 0.5
	            parameters['agent_based']['max_daily_vax'] = 50
	        if i == 1: 
	            parameters['system_dynamics']['quarantine_fraction'] = 0.9
	            parameters['agent_based']['max_daily_vax'] = 50
	        if i == 2:
	            parameters['system_dynamics']['quarantine_fraction'] = 0.9
	            parameters['agent_based']['max_daily_vax'] = 1000
	    
	        # Store maximum no. of infections for each replication
			# We'll use parallel processing to speed things up
	        loop_results = Parallel(n_jobs=cpu_count())(
	            delayed(run_hybrid_model)(parameters, time_domain, j)
	            for j in range(N_REPLICATIONS)
	        )
	        results[i] = np.mean(loop_results)
	
	    results_dict[method] = results

	# Plot the results
	x = np.arange(len(scenarios))
	width = 0.4 
	multiplier = 0.5
	fig, ax = plt.subplots(figsize = (10,4), layout='constrained')
	labels = {
	    'interp': 'Interpolation',
	    'LCT': 'Erlang'
	}
	colours = {
	    'interp': 'darkcyan',
	    'LCT': 'darkorange'
	}
	for method, value in results_dict.items():
	    offset = width * multiplier
	    rects = ax.bar(x + offset, np.round(value), width, 
	                   label=labels[method], color=colours[method])
	    ax.bar_label(rects, padding=3)
	    multiplier += 1
	ax.set_ylabel('Peak number of infections', fontsize=12)
	ax.set_xticks(x + width, scenarios)
	ax.legend(loc='upper right', ncols=2, fontsize=12)
	ax.set_ylim(0, 4000)
	plt.xticks(fontsize=11)
	plt.yticks(fontsize=11)
	plt.show()
	fig.savefig('../figures/peak-infections-plt.png', dpi=300)

	# Print the total run time
	main_end = time.time()
	main_elapsed = main_end - main_start
	print(f'Total run time: {np.round(main_elapsed/3600, decimals=1)} hours.')

if __name__ == '__main__':
	main()