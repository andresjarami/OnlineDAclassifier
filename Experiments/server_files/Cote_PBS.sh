#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=30
#PBS -l mem=3GB
#PBS -l walltime=20:00:00
#PBS -l software=python
#PBS -l wd

place="../Results/"
database="Cote"
printResults=0
shotStart=1
for featureSet in {1..3}
do
  for expTime in {1..20}
  do
    for person in {20..36}
    do
    python3 ../main_exp.py $featureSet $person $person $place $database $printResults $shotStart $expTime $expTime > logs/Cote$person$featureSet$PBS_JOBID.log &
    done
  done
done
wait
