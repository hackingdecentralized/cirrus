#!/bin/bash

# Define ranges for num-vars and log-num-workers
num_vars_values=(1 2 4 8 16 32)
output_log="benchmark_times_no_ha.log"

# Clear the log file if it exists
> "$output_log"

# Loop through each value of num-vars
for num_vars in "${num_vars_values[@]}"; do
  # Calculate the maximum value for log-num-workers based on num-vars
  max_log_num_workers=16

  # Loop through each value of log-num-workers from 1 to min(num_vars, 20)
  for ((log_num_workers = 1; log_num_workers <= max_log_num_workers; log_num_workers++)); do
    # Run the cargo command with the current values and capture the output
    echo "Running benchmark with --num-vars=$((num_vars)) and --log-num-workers=$log_num_workers"
    output=$(cargo run --release --package cirrus --bin bench_master --features bench-master -- \
      --num-vars "$((num_vars + log_num_workers))" --log-num-workers "$log_num_workers" --num-threads 1 2>&1)

    # Extract the elapsed time and normalize units
    raw_time=$(echo "$output" | grep -oP '(?<=time elapsed: )[\d.]+(?:ms|s)')

    # Check if time is in seconds or milliseconds, and normalize to milliseconds
    if [[ $raw_time == *"s" && $raw_time != *"ms" ]]; then
      # Convert seconds to milliseconds by removing "s" and multiplying by 1000
      time_in_ms=$(echo "$raw_time" | sed 's/s//' | awk '{printf "%.3f", $1 * 1000}')
    else
      # Time is already in milliseconds, remove "ms" for consistency
      time_in_ms=${raw_time/ms/}
    fi

    # Log the result with parameters and normalized elapsed time
    echo "--num-vars=$num_vars, --log-num-workers=$log_num_workers, time elapsed: ${time_in_ms}ms" >> "$output_log"

    # Optionally, display the elapsed time for each run
    echo "Time elapsed for --num-vars=$num_vars and --log-num-workers=$log_num_workers: ${time_in_ms}ms"
  done
done

echo "Benchmarking complete. Results saved to $output_log"