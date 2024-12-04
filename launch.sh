#! /bin/bash

logdir=$1
echo "Starting runs in logdir: $2"
mkdir $logdir

for run_num in {0..15}; do
    # Redirect each output of nohup to a file and the error to another file
    nohup python main.py --logdir=$logdir --run-num=$run_num > ${logdir}/${run_num}-out.nohup 2> ${logdir}/${run_num}-error.nohup &
done

echo "All runs started"