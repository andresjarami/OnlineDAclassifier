#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=48
#PBS -l mem=15GB
#PBS -l walltime=48:00:00
#PBS -l software=python
#PBS -l wd

place="../Results/"
database="EPN_120"
printResults=0
shotStart=1
for featureSet in {1..3}
do
  for expTime in {1..20}
  do
    for person in {1..120}
    do
    python3 ../main_exp.py $featureSet $person $person $place $database $printResults $shotStart $expTime $expTime > logs/EPN$person$featureSet$PBS_JOBID.log &
    done
  done
done
wait
