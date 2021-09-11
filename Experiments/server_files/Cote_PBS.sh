#!/bin/bash
#PBS -P oj72
#PBS -q normal
#PBS -l ncpus=30
#PBS -l mem=3GB
#PBS -l walltime=1:00:00
#PBS -l software=python
#PBS -l wd

place="final3/"
database="Cote"
printResults=0
shotStart=1
featureSet=1
for expTime in {2..2}
do
	for person in {20..36}
	do
	python3 main_exp.py $featureSet $person $person $place $database $printResults $shotStart $expTime $expTime > logs/Cote$person$featureSet$PBS_JOBID.log &
done
done
wait
