#!/bin/bash
trap ctrl_c INT
function ctrl_c() {
    echo "Script interrupted. Exiting..."
    exit 1
}

rm logs.txt

for agent in "rppo" "ppo" "vae" "random"; do
    echo "Running experiments with retraining for $agent"
    python src/main.py --agent=$agent --anomaly --n_jobs=10 --n_repetitions=10 --retrain_interval=30 --initial_seed=100
    python src/main.py --agent=$agent --anomaly --n_jobs=10 --n_repetitions=10 --retrain_interval=30 --initial_seed=110
    python src/main.py --agent=$agent --anomaly --n_jobs=10 --n_repetitions=10 --retrain_interval=30 --initial_seed=120
done

exit 0


for anomaly in "--anomaly" ""; do
    for modification in "" "--modification"; do
        for agent in "rppo" "ppo" "vae"; do
            if [[ "$agent" != "random" ]]; then
                for i in {1..5}; do
                    python src/agents_tuning.py --agent=$agent $anomaly $modification --n_trials=100 --n_runs=5 &
                done
                echo "Waiting for tuning of $agent $anomaly $modification to complete..."
                wait
            fi
            echo "Tuning of $agent $anomaly $modification completed."
        done
    done
done
