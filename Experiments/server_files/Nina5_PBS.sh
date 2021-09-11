#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=30
#PBS -l mem=3GB
#PBS -l walltime=20:00:00
#PBS -l software=python
#PBS -l wd

place="final3/"
database="Nina5"
printResults=0
shotStart=1
featureSet=1
for expTime in {2..2}
do
	for person in {1..10}
	do
	python3 main_exp.py $featureSet $person $person $place $database $printResults $shotStart $expTime $expTime > logs/Nina5$person$featureSet$PBS_JOBID.log &
done
done
wait
