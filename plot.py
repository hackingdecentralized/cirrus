import matplotlib.pyplot as plt
import numpy as np
import re

# Initialize data dictionary to store parsed data
data = {}

# Read data from file
file_path = 'benchmark_times.log'
with open(file_path, 'r') as file:
    for line in file:
        # Extract num_vars, log_num_workers, and time
        match = re.search(r'--num-vars=(\d+), --log-num-workers=(\d+), time elapsed: ([\d.]+)ms', line)
        if match:
            num_vars = int(match.group(1))
            log_num_workers = int(match.group(2))
            time_elapsed = float(match.group(3))
            
            # Store data organized by num_vars
            if num_vars not in data:
                data[num_vars] = {'log_num_workers': [], 'time_elapsed': []}
            data[num_vars]['log_num_workers'].append(log_num_workers)
            data[num_vars]['time_elapsed'].append(time_elapsed)

# Plotting
plt.figure(figsize=(10, 6))
for num_vars, values in data.items():
    # Plot with log scale on y-axis
    plt.plot(values['log_num_workers'], values['time_elapsed'], label=f'num-vars={num_vars}')

# Set plot labels and title
plt.xlabel('log_num_workers')
plt.ylabel('Log(Time Elapsed in ms)')
# Set x-axis to integer values from 1 to 19
plt.xticks(ticks=range(1, 20), labels=range(1, 20))
# Set y-axis to log scale
plt.yscale('log', base=2)
y_ticks = [2**i for i in range(3, 15)]
y_tick_labels = [f'$2^{{{i}}}$' for i in range(3, 15)]
plt.yticks(y_ticks, y_tick_labels)
plt.title('Benchmark Times by num-vars and log_num_workers')
plt.legend(title='num-vars')
plt.grid(True)
plt.savefig("benchmark_plot.png")