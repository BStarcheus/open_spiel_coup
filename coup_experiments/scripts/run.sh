#!/usr/bin/env bash

algos=("deep_cfr" "deep_cfr_tf2" "nfsp" "mccfr")

cd /repo/coup_experiments/data
for algo in "${algos[@]}"; do
    if [ ! -d "${algo}" ]; then
        mkdir ${algo}
    fi
done

cd /repo/coup_experiments/scripts
for algo in "${algos[@]}"; do
    for flagfile in flags/${algo}-*.cfg; do
        [ -f "${flagfile}" ] || continue
        echo "Starting ${algo} on ${flagfile}"
        python3 ${algo}.py --flagfile=${flagfile}
    done
done