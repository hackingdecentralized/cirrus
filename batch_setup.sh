#!/bin/bash

# Specify the range for log_num_workers and log_num_vars
log_num_workers_start=8  # Starting value for log_num_workers
log_num_workers_end=8    # Ending value for log_num_workers

log_num_vars_start=24    # Starting value for log_num_vars
log_num_vars_end=24      # Ending value for log_num_vars

curve="bls12_381"        # Specify the curve to use

# Loop through the range of log_num_workers
for (( log_num_workers=$log_num_workers_start; log_num_workers<=$log_num_workers_end; log_num_workers++ ))
do
    # Loop through the range of log_num_vars
    for (( log_num_vars=$log_num_vars_start; log_num_vars<=$log_num_vars_end; log_num_vars++ ))
    do
        echo "Running ./scripts/run.sh setup --log_num_workers $log_num_workers --log_num_vars $log_num_vars --curve $curve"
        ./scripts/run.sh setup --log_num_workers $log_num_workers --log_num_vars $log_num_vars --curve $curve
    done
done

echo "Batch setup complete."