#!/bin/bash
trap ctrl_c INT
function ctrl_c() {
    echo "Script interrupted. Exiting..."
    exit 1
}



for anomaly in "--anomaly" ""; do
    for modification in "--modification" ""; do
        for agent in "rppo" "ppo" "vae" "random"; do
            if [[ "$agent" != "random" ]]; then
                for i in {1..6}; do
                    python src/agents_tuning.py --agent=$agent $anomaly $modification --n_trials=100 --n_runs=5 &
                done
                echo "Waiting for tuning of $agent $anomaly $modification to complete..."
                wait
            fi
            echo "Tuning of $agent $anomaly $modification completed."
            # echo "Starting experiments..."
            # python src/main.py --agent=$agent $anomaly $modification --n_jobs=30 --n_repetitions=30
        done
    done
done
