import matplotlib.pyplot as plt
import numpy as np
import re

# Initialize data dictionary to store parsed data
data = {}

# Read data from file
file_path = 'benchmark_times_1.log'
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
log_num_workers = []
time = []
for num_vars, values in data.items():
    if num_vars <= 2:
        log_num_workers = values['log_num_workers']
        if len(time) == 0:
            time = np.array(values['time_elapsed'])
        else:
            time = time + np.array(values['time_elapsed'])
        continue
    if num_vars < 8:
        continue
    # Plot with log scale on y-axis
    plt.plot(values['log_num_workers'], values['time_elapsed'], label=f'w/o LB, log (T) = {num_vars}')

plt.plot(values['log_num_workers'], time/2, label=f'w/ LB')

# Set plot labels and title
plt.xlabel('log_num_workers')
plt.ylabel('Log(Time Elapsed in ms)')
# Set x-axis to integer values from 1 to 19
plt.xticks(ticks=range(1, 17), labels=range(1, 17))
# Set y-axis to log scale
# plt.yscale('log', base=2)
# y_ticks = [2**i for i in range(1, 15)]
# y_tick_labels = [f'$2^{{{i}}}$' for i in range(1, 15)]

y_ticks = [0] + [2**i for i in range(9, 15)]
y_tick_labels = ["0"] + [f'$2^{{{i}}}$' for i in range(9, 15)]
plt.yticks(y_ticks, y_tick_labels)
plt.xlim(1, 16)
plt.ylim(0, 2**14)
plt.title('Benchmark Times by num-vars and log_num_workers')
plt.legend()
plt.grid(True)
plt.savefig("benchmark_plot.png")
plt.savefig("benchmark_plot.pdf")