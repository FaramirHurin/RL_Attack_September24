#!/bin/bash
trap ctrl_c INT
function ctrl_c() {
    echo "Script interrupted. Exiting..."
    exit 1
}

ps_data=$(ps -fp 1372)
echo "Waiting for process with PID 1372: $ps_data"
while kill -0 1372 2>/dev/null; do
    sleep 1s
done
echo "Process with PID 1372 has completed!"


rm logs.txt
python src/main.py --agent=vae --know_client --n_jobs=30 --n_repetitions=30
python src/classifier-tuning.py 
exit 0


for anomaly in "" "--anomaly"; do
    for modification in "" "--modification"; do
        for agent in "vae" "rppo" "ppo" ; do
            if [[ "$agent" == "vae" ]]; then
                know_cient="--know_client"
            else
                know_cient=""
            fi
            if [[ "$agent" != "random" ]]; then
                for i in {1..6}; do
                    python src/agents_tuning.py --agent=$agent $anomaly $modification $know_cient --n_trials=100 --n_runs=5 &
                done
                echo "Waiting for tuning of $agent $anomaly $modification to complete..."
                wait
            fi
            echo "Tuning of $agent $anomaly $modification completed."
            python src/main.py --agent=$agent $anomaly $modification $know_cient --n_jobs=30 --n_repetitions=30
        done
    done
done
