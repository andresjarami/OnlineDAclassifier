#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=30
#PBS -l mem=4GB
#PBS -l walltime=48:00:00
#PBS -l software=python
#PBS -l wd

place="../Results/"
database="Capgmyo_dbb"
printResults=0
shotStart=1
for featureSet in {1..3}
do
  for expTime in {1..20}
  do
    for person in {1..10}
    do
    python3 ../main_exp.py $featureSet $person $person $place $database $printResults $shotStart $expTime $expTime > logs/Capgmyo_dbb$person$featureSet$PBS_JOBID.log &
    done
  done
done
wait
