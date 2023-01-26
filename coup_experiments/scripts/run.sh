#!/usr/bin/env bash

algos=("deep_cfr" "deep_cfr_tf2" "nfsp" "mccfr")

cd /repo/coup_experiments/data
for algo in "${algos[@]}"; do
    if [ ! -d "${algo}" ]; then
        mkdir ${algo}
    fi
done

cd /repo/coup_experiments/scripts
num_run_parallel=2
ind=0
for algo in "${algos[@]}"; do
    for flagfile in flags/${algo}*.cfg; do
        [ -f "${flagfile}" ] || continue
        ind=$((ind+1))
        echo "Starting ${algo} on ${flagfile}"
        python3 ${algo}.py --flagfile=${flagfile} &
        if [ $((ind%num_run_parallel)) -eq 0 ]; then
            wait
        fi
    done
done