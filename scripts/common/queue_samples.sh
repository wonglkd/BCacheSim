#!/bin/bash
TRACE=$1
TRACE_GROUP=$2
QUEUE=par2

for rate in 0.0001 0.0005 0.001 0.01 0.05 0.1; do
    for (( i = 0; i < 10; i++ )); do
        start=$(python -c "print('{:g}'.format($i * $rate))")
        if [[ $i -ge 1 && $rate == 0.2 ]]; then
            continue
        fi
        if [[ $i -ge 1 && $rate == 0.1 ]]; then
            continue
        fi
        if [[ -f ~/fb/ws/${TRACE_GROUP}/${TRACE}/full_${start}_{$rate}.trace ]]; then
            continue
        fi
        ~/ws_traces/episodic_analysis/local_runcmd.py --job-id "samplepy__${TRACE}_full_${start}_${rate}" --queue ${QUEUE} --timeout 1800 \
            "cd ~/ws_traces/scripts/mar2023/; ./sample_trace.sh ${TRACE_GROUP} ${TRACE} full -l ${rate} --start ${start}"
    done
done
