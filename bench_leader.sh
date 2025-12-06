#!/usr/bin/env bash
set -euo pipefail

t_values=(8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
output_log="benchmark_times_leader_logT.log"

# Clean log
: > "$output_log"

compute_log_m_prime() {
  local t="$1"
  if (( t <= 8 )); then
    echo 3
  elif (( t <= 16 )); then
    echo 4
  else
    echo 5
  fi
}

for T in "${t_values[@]}"; do
  log_num_workers="$(compute_log_m_prime "$T")"

  echo "Running benchmark with --num-vars=${T} and --log-num-workers=${log_num_workers}"
  output=$(cargo run --release --package cirrus --bin bench_master --features bench-master -- \
    --num-vars "$((T + log_num_workers))" \
    --log-num-workers "${log_num_workers}" \
    --num-threads 8 2>&1)

  # Parse "time elapsed: <number>(ms|s)"
  raw_time=$(echo "$output" | grep -oP '(?<=time elapsed: )[\d.]+(?:ms|s)' || true)

  if [[ -z "$raw_time" ]]; then
    echo "WARNING: Could not parse elapsed time for T=${T}, logW=${log_num_workers}" | tee -a "$output_log"
    echo "$output" >> "$output_log"
    continue
  fi

  # Normalize to ms
  if [[ "$raw_time" == *"s" && "$raw_time" != *"ms"* ]]; then
    time_in_ms=$(echo "$raw_time" | sed 's/s//' | awk '{printf "%.3f", $1 * 1000}')
  else
    time_in_ms=${raw_time/ms/}
  fi

  # Log results
  echo "--num-vars=${T}, time elapsed: ${time_in_ms}ms" | tee -a "$output_log"
  echo "Time elapsed for --num-vars=${T}: ${time_in_ms}ms"
  sleep 1
done

echo "Benchmarking complete. Results saved to ${output_log}"