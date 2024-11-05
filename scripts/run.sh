#!/bin/bash

# Default values for the parameters
gate="vanilla"
log_num_vars=15
log_num_workers=3
num_threads=8
fdir=""
username="ubuntu"
worker_id=0
master_addr="127.0.0.1:8000"

# Function to display usage instructions
usage() {
    echo "Usage: $0 {setup|run_master|run_multi_worker|run_single_worker|analyze} [options]"
    echo "Options:"
    echo "  --gate <gate>                - The gate type (default: vanilla)"
    echo "  --log_num_vars <log_num_vars>        - The number of variables (default: 15)"
    echo "  --log_num_workers <log_num_workers> - Logarithmic number of workers (default: 3)"
    echo "  --num_threads <num_threads>  - Number of threads (default: 8)"
    echo "  --fdir <fdir>                - Output directory"
    echo "  --worker_id <worker_id>      - Worker ID for single worker process (default: 0)"
    echo "  --username <username>        - Username for memory monitoring (default: ubuntu)"
    echo "  --master_addr <master_addr>  - Master address (default:127.0.0.1:8000)"
    echo "Commands:"
    echo "  setup             - Sets up the directories and runs the setup command."
    echo "  run_master        - Starts the master process."
    echo "  run_multi_worker  - Starts multiple worker processes."
    echo "  run_single_worker - Starts a single worker process."
    echo "  analyze           - Analyzes a log file. Usage: analyze <log_file> <output_file>"
    exit 1
}

# Parse command-line arguments for options
action=$1
shift # Shift to process remaining options

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gate)
            gate="$2"
            shift 2
            ;;
        --log_num_vars)
            log_num_vars="$2"
            shift 2
            ;;
        --log_num_workers)
            log_num_workers="$2"
            shift 2
            ;;
        --num_threads)
            num_threads="$2"
            shift 2
            ;;
        --fdir)
            fdir="$2"
            shift 2
            ;;
        --worker_id)
            worker_id="$2"
            shift 2
            ;;
        --username)
            username="$2"
            shift 2
            ;;
        --master_addr)
            master_addr="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Set default fdir if not specified
if [[ -z "$fdir" ]]; then
    fdir="out/${gate}-${log_num_vars}-${log_num_workers}"
fi

# Calculate the number of workers
num_workers=$((2**log_num_workers))

# Setup function to create directories and run setup command
setup() {
    echo "Running setup..."
    if [ ! -d "$fdir" ]; then
        mkdir -p "$fdir"
        echo "Directory $fdir created"
    else
        echo "Directory $fdir already exists"
    fi

    # Run the setup command with cargo
    cargo run --bin setup -- --output "$fdir" --log-num-constraints "$log_num_vars" --log-num-workers "$log_num_workers"
}

# Function to run the master process
run_master() {
    echo "Starting master process..."
    cargo run --bin master --features print-trace -- --num-threads "$num_threads" --circuit-file "$fdir/circuit.plonk" --pk-master "$fdir/master.pk" --verification-key "$fdir/verify.key" --master-addr "$master_addr" > "$fdir/master.log" &
    master_pid=$!

    # Start top in the background to monitor memory usage, filtering by the master process
    top -d 1 -b -u "$username" | grep -w "master" > "$fdir/master_memory.log" &
    top_pid=$!

    # Loop to check if the master process is still running
    while kill -0 "$master_pid" 2>/dev/null; do
        sleep 1
    done

    # Kill the specific top process and its grep subprocess if they are still running
    if kill -0 "$top_pid" 2>/dev/null; then
        pkill -P "$top_pid"   # Kill any child processes (like grep) started by top
        kill "$top_pid"       # Kill the top process itself
        echo "Monitoring for master process terminated."
    fi
}

# Function to run multiple workers
run_multi_worker() {
    echo "Starting multi-worker processes..."
    # Array to store PIDs of worker processes
    # worker_pids=()
    
    # Loop to start each worker
    for ((i=0; i<num_workers-1; i++)); do
        cargo run --bin worker -- --num-threads "$num_threads" --worker-id "$i" --pk-worker "$fdir/worker_$i.pk" --master-addr "$master_addr" &
        # worker_pids+=($!)  # Store each worker's PID
    done

    # Start the final worker with logging
    # i=$((num_workers - 1))
    cargo run --bin worker --features print-trace -- --num-threads "$num_threads" --worker-id "$i" --pk-worker "$fdir/worker_$i.pk" --master-addr "$master_addr" > "$fdir/worker.log" &
    # worker_pids+=($!)

    # # Start top in the background to monitor memory usage, filtering by worker processes
    # top -d 1 -b -u "$username" | grep -w "worker" > "$fdir/worker_memory.log" &
    # top_pid=$!

    # # Loop to check if any worker processes are still running
    # while true; do
    #     all_done=true
    #     for pid in "${worker_pids[@]}"; do
    #         if kill -0 "$pid" 2>/dev/null; then
    #             all_done=false
    #             break
    #         fi
    #     done

    #     # Exit the loop if all workers are done
    #     if $all_done; then
    #         break
    #     fi

    #     sleep 1
    # done

    # # Stop top once all worker processes terminate
    # kill "$top_pid"
}

# Function to run a single worker
run_single_worker() {
    echo "Starting single worker process..."

    # Run a single worker with ID 0 and capture its PID
    cargo run --bin worker --features print-trace -- --num-threads "$num_threads" --worker-id "$worker_id" --pk-worker "$fdir/worker_$worker_id.pk" --master-addr "$master_addr" > "$fdir/worker_$worker_id.log" &
    single_worker_pid=$!

    # Start top in the background to monitor memory usage for this specific worker
    top -d 1 -b -u "$username" | grep -w "worker" > "$fdir/worker_${worker_id}_memory.log" &
    top_pid=$!

    # Wait for the single worker process to terminate
    while kill -0 "$single_worker_pid" 2>/dev/null; do
        sleep 1
    done

    # Terminate the specific top and its child (grep) once the worker is done
    if kill -0 "$top_pid" 2>/dev/null; then
        pkill -P "$top_pid"   # Kill any child processes (like grep) started by top
        kill "$top_pid"       # Kill the top process itself
        echo "Monitoring for single worker process terminated."
    fi
}

# Function to analyze a given log file
analyze() {
    log_file=$fdir/"$1".log
    output_file=$fdir/$1_analysis.json

    if [[ -z "$log_file" || -z "$output_file" ]]; then
        echo "Error: Please provide a log file and an output file for analysis."
        echo "Usage: $0 analyze <log_file> <output_file>"
        exit 1
    fi

    echo "Analyzing log file $log_file..."
    python3 analyze.py -l "$log_file" -o "$output_file"
    echo "Analysis complete. Results saved to $output_file"
}

# Main script entry point
if [[ -z "$action" ]]; then
    usage
fi

# Execute the action based on the command argument
case "$action" in
    setup)
        setup
        ;;
    run_master)
        run_master
        ;;
    run_multi_worker)
        run_multi_worker
        ;;
    run_single_worker)
        run_single_worker
        ;;
    analyze)
        analyze "$2" "$3"
        ;;
    *)
        usage
        ;;
esac
