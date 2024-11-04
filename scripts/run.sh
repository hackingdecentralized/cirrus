#!/usr/bin/env bash
gate=vanilla
num_vars=20
log_num_workers=3
num_workers=$((2**$log_num_workers))
num_threads=4
fdir=$gate\_$num_vars\_$log_num_workers

mkdir -p $fdir
cargo run --bin setup -- --output $fdir --log-num-constraints $num_vars --log-num-workers $log_num_workers
cargo run --bin master --features print-trace -- --num-threads $num_threads --circuit-file $fdir/circuit.plonk --pk-master $fdir/master.pk --verification-key $fdir/verify.key > $fdir/master.log

for i in $(seq 0 $((num_workers-2))); do
    cargo run --bin worker -- --num-threads $num_threads --worker-id $i --pk-worker $fdir/worker_$i.pk &
done

i=$((num_workers-1))
cargo run --bin worker --features print-trace -- --num-threads $num_threads --worker-id $i --pk-worker $fdir/worker_$i.pk > $fdir/worker.log

# memory usage of a worker
top -d 1 -b | grep --line-buffered worker > memory.log
