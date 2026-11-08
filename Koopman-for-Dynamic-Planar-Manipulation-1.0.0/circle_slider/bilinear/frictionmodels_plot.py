# Create plot(s) for the different friction results
# The data is stored in a list of dictionaries, where each dictionary contains the results for a different initial condition and friction coefficient.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    # Load the data
    var_mu_data = np.load('variable_mu/results/variable_mu_results2.npy', allow_pickle=True)
    og_data = np.load('results/og_fric_results.npy', allow_pickle=True)

    # Go through each dictionary and add a 'sum_cost' key
    for entry in var_mu_data:
        entry['sum_cost'] = np.sum(entry['poserrsq_data'])
        entry['mpc_sum_cost'] = np.sum(entry['MPC_cost_data'])
        if entry['sum_cost']>= 2.5 and entry['mu_t'] == 10.:
            print("High cost detected:", entry['sum_cost'], "for mu_t:", entry['mu_t'])
            print("Init config:", entry['init_config'])

    for entry in og_data:
        entry['sum_cost'] = np.sum(entry['poserrsq_data'])
        entry['mpc_sum_cost'] = np.sum(entry['MPC_cost_data'])

    # Convert the data to a DataFrame
    var_mu_df = pd.DataFrame(list(var_mu_data))
    og_df = pd.DataFrame(list(og_data))
    var_mu_df['friction'] = 'Robust'
    og_df['friction'] = 'Original'
    combined_df = pd.concat([var_mu_df, og_df], ignore_index=True)

    # Create a boxplot for the cost
    plt.figure(figsize=(9, 5.4))
    sns.boxplot(x='mu_t', y='sum_cost', hue='friction', data=combined_df)
    plt.title('Cost Comparison: Robust vs Original Models')
    plt.savefig('results/variable_mu_cost_comparison.png', dpi=300)
    plt.show()
