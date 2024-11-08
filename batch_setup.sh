#!/bin/bash

# Specify the range for log_num_workers and log_num_vars
log_num_workers_start=1  # Starting value for log_num_workers
log_num_workers_end=3    # Ending value for log_num_workers

log_num_vars_start=25    # Starting value for log_num_vars
log_num_vars_end=25      # Ending value for log_num_vars

# Loop through the range of log_num_workers
for (( log_num_workers=$log_num_workers_start; log_num_workers<=$log_num_workers_end; log_num_workers++ ))
do
    # Loop through the range of log_num_vars
    for (( log_num_vars=$log_num_vars_start; log_num_vars<=$log_num_vars_end; log_num_vars++ ))
    do
        echo "Running ./scripts/run.sh setup --log_num_workers $log_num_workers --log_num_vars $log_num_vars"
        ./scripts/run.sh setup --num_threads 25 --log_num_workers $log_num_workers --log_num_vars $log_num_vars
    done
done

echo "Batch setup complete."